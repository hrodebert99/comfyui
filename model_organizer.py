import sqlite3
from pathlib import Path
import os
import datetime
import hashlib
import requests
import time

DATABASE_PATH = Path('database.db')

SQL_CREATE_TABLES = '''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE NOT NULL,
        hash TEXT UNIQUE NOT NULL,
        civitai_model_id INTEGER UNIQUE REFERENCES civitai_models (id) DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS civitai_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type_id INTEGER REFERENCES types (id) NOT NULL,
        base_model_id INTEGER REFERENCES base_models (id) NOT NULL,
        creator_id INTEGER REFERENCES creators (id) DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS base_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS creators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
'''

MODEL_PATH = Path('models')
MODEL_EXTENSIONS = [
    '.safetensors'
]
MODEL_TYPE_PATHS = [
    MODEL_PATH.joinpath('checkpoints'),
    MODEL_PATH.joinpath('embeddings'),
    MODEL_PATH.joinpath('loras'),
    MODEL_PATH.joinpath('vae')
]
MODEL_NOT_FOUND_PATH = MODEL_PATH.joinpath('loras', 'null')
MODEL_TYPE_TO_DIRECTORY_MAP = {
    'Checkpoint': 'checkpoints',
    'TextualInversion': 'embeddings',
    'LORA': 'loras',
    'LoCon': 'loras',
    'DoRA': 'loras',
    'VAE': 'vae'
}

LOG_FILENAME = f'{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log'

REQUEST_INTERVAL = 60

def connect_to_database():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    return connection, cursor

def get_all_models_path():
    paths: list[Path] = []

    for dirpath, _, filenames in os.walk(MODEL_PATH):
        for filename in filenames:
            paths.append(Path(dirpath, filename))
    
    return paths

def insert_into_database(cursor: sqlite3.Cursor, connection: sqlite3.Connection, table: str, columns: dict, enable_exception = False, commit = False):
    sql = f'INSERT INTO {table} ({', '.join(columns.keys())}) VALUES ({', '.join(['?'] * len(columns.keys()))})'
    parameters = tuple(columns.values())
    
    try:
        cursor.execute(sql, parameters)
    except sqlite3.IntegrityError as exception:
        if enable_exception:
            log(exception)
            exit()
        else:
            pass

    if commit:
        connection.commit()

def log(message):
    message = f'[{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] {message}'
    with open(LOG_FILENAME, 'a', encoding = 'utf-8') as file:
        file.write(f'{message}\n')
    print(message)

def calculate_hash(filepath: Path):
    hasher = hashlib.sha256()
    with filepath.open('rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def send_get_request(url):
    response = requests.get(url)
    while response.status_code == 500:
        time.sleep(REQUEST_INTERVAL)
        response = requests.get(url)
    return response

if __name__ == '__main__':
    connection, cursor = connect_to_database()

    cursor.executescript(SQL_CREATE_TABLES)

    insert_into_database(cursor, connection, 'creators', { 'name': 'null' }, commit = True)

    for model_path in get_all_models_path():
        if model_path.suffix not in MODEL_EXTENSIONS:
            continue
        if Path(*model_path.parts[:2]) not in MODEL_TYPE_PATHS:
            continue
        log(model_path)

        cursor.execute('SELECT 1 FROM models WHERE path = ?', (str(model_path), ))
        if cursor.fetchone() != None:
            continue
        
        model_hash = calculate_hash(model_path)

        cursor.execute('SELECT 1 FROM models WHERE hash = ?', (model_hash, ))
        if cursor.fetchone() != None:
            continue

        if model_path.parent == MODEL_NOT_FOUND_PATH:
            insert_into_database(cursor, connection, 'models', { 'path': str(model_path), 'hash': model_hash })
        else:
            model_type, model_base_model, model_creator = model_path.parts[:3]
            
            response = send_get_request(f'https://civitai.com/api/v1/model-versions/by-hash/{model_hash}')
            if response.status_code == 404:
                model_type = {value: key for key, value in MODEL_TYPE_TO_DIRECTORY_MAP.items() if key not in ['LoCon', 'DoRA']}.get(model_type)
            else:
                model_type = response.json()['model']['type']

            insert_into_database(cursor, connection, 'types', {'name': model_type})
            cursor.execute('SELECT id FROM types WHERE name = ?', (model_type, ))
            model_type_id = dict(cursor.fetchone())['id']

            insert_into_database(cursor, connection, 'base_models', {'name': model_base_model})
            cursor.execute('SELECT id FROM base_models WHERE name = ?', (model_base_model, ))
            model_base_model_id = dict(cursor.fetchone())['id']

            insert_into_database(cursor, connection, 'creators', {'name': model_creator})
            cursor.execute('SELECT id FROM creators WHERE name = ?', (model_creator, ))
            model_creator_id = dict(cursor.fetchone())['id']

            insert_into_database(cursor, connection, 'civitai_models', {'type_id': model_type_id, 'base_model_id': model_base_model_id, 'creator_id': model_creator_id})
            model_civitai_model_id = cursor.lastrowid

            cursor.execute('INSERT INTO models (path, hash, civitai_model_id) VALUES (?, ?, ?)', (str(model_path), model_hash, model_civitai_model_id))

        connection.commit()

    for model_path in get_all_models_path():
        if model_path.suffix not in MODEL_EXTENSIONS:
            continue
        if model_path.parent != MODEL_PATH:
            continue
        log(model_path)

        model_hash = calculate_hash(model_path)

        cursor.execute('SELECT 1 FROM models WHERE hash = ?', (model_hash, ))
        if cursor.fetchone() != None:
            continue

        response = send_get_request(f'https://civitai.com/api/v1/model-versions/by-hash/{model_hash}')
        if response.status_code == 404:
            new_model_path = MODEL_NOT_FOUND_PATH.joinpath(model_path.name)
        else:
            model_version = response.json()

            response = send_get_request(f'https://civitai.com/api/v1/models/{model_version['modelId']}')
            model = response.json()
        
            model_type = model_version['model']['type']
            model_base_model = model_version['baseModel']
            if 'creator' in model:
                model_creator = model['creator']['username']
            else:
                model_creator = 'null'

            new_model_path = Path(MODEL_PATH, MODEL_TYPE_TO_DIRECTORY_MAP.get(model_type), model_base_model, model_creator, model_path.name)

            insert_into_database(cursor, connection, 'types', {'name': model_type})
            cursor.execute('SELECT id FROM types WHERE name = ?', (model_type, ))
            model_type_id = dict(cursor.fetchone())['id']

            insert_into_database(cursor, connection, 'base_models', {'name': model_base_model})
            cursor.execute('SELECT id FROM base_models WHERE name = ?', (model_base_model, ))
            model_base_model_id = dict(cursor.fetchone())['id']

            insert_into_database(cursor, connection, 'creators', {'name': model_creator})
            cursor.execute('SELECT id FROM creators WHERE name = ?', (model_creator, ))
            model_creator_id = dict(cursor.fetchone())['id']

            insert_into_database(cursor, connection, 'civitai_models', {'type_id': model_type_id, 'base_model_id': model_base_model_id, 'creator_id': model_creator_id}, True)
            model_civitai_model_id = cursor.lastrowid

            insert_into_database(cursor, connection, 'models', {'path': str(new_model_path), 'hash': model_hash, 'civitai_model_id': model_civitai_model_id}, True)

        if new_model_path.exists():
            connection.rollback()
            continue

        new_model_path.parent.mkdir(parents = True, exist_ok = True)
        model_path.rename(new_model_path)
        log(f'    Moved into {new_model_path}')
            
        connection.commit()
