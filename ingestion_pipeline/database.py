"""
This file loads csvs into the database and extracts data from the database
"""

import pandas as pd
import sqlite3

def load_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df['id'] = df.index
    return df

def csv_to_sqlite(csv_file: str, db_file: str, table_name: str):
    df = pd.read_csv(csv_file)
    df['id'] = df.index
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
    conn.close()
    print(f"{df.shape[0]} stored into {table_name} in {db_file}")

def store_to_sqlite(df: pd.DataFrame, db_file: str, table_name: str):
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
    conn.close()
    print(f"{df.shape[0]} stored into {table_name} in {db_file}")

def get_df(table_name: str, db_file: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=conn)
    return df

def get_texts(table_name: str, texts_col_name: str, db_file: str) -> list[str]:
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=conn)
    texts = df[f"{texts_col_name}"].tolist()
    return texts

def get_texts_and_metadata(table_name: str, texts_col_name: str, db_file: str) -> list[str]:
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=conn)
    texts = df[f"{texts_col_name}"].tolist()
    return texts, df
