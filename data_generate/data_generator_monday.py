# data_generator.py
import os
import time
from argparse import ArgumentParser

import pandas as pd
import psycopg2

def get_data(dataset):
    file_path = f'{dataset}_monday.json'
    df = pd.read_json(file_path, orient='records', lines=True)
    return df

def create_table(db_connect, dataset):
    drop_table_query = f"DROP TABLE IF EXISTS {dataset};"
    create_table_query = f"""
    CREATE TABLE {dataset} (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        image_path varchar(255),
        target int
    );
    """
    
    with db_connect.cursor() as cur:
        cur.execute(drop_table_query)
        cur.execute(create_table_query)
        db_connect.commit()

def insert_dataframe(db_connect, dataset, df):
    for index, row in df.iterrows():
        insert_row_query = f"""
        INSERT INTO {dataset}
            (timestamp, image_path, target)
            VALUES (
                NOW(),
                '{row['image_path']}',
                {row['target']}
            );
        """
        with db_connect.cursor() as cur:
            cur.execute(insert_row_query)
            db_connect.commit()

def insert_data(db_connect, dataset, data):
    insert_row_query = f"""
    INSERT INTO {dataset}
        (timestamp, image_path, target)
        VALUES (
            NOW(),
            '{data.image_path}',
            {data.target}
        );
    """
    print("run")
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        print(insert_row_query)
        db_connect.commit()


def generate_data(db_connect, dataset, df):
    insert_dataframe(db_connect, dataset, df)
    while True:
        insert_data(db_connect, dataset, df.sample(1).squeeze())
        time.sleep(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    args = parser.parse_args()

    dataset = os.getenv("DATASET")

    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host=args.db_host,
        port=5432,
        database="mydatabase",
    )
    create_table(db_connect, dataset)
    df = get_data(dataset)
    generate_data(db_connect, dataset, df)