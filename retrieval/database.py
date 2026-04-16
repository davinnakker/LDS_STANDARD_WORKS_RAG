"""
This file loads csvs into the database and extracts data from the database
"""
import pandas as pd
import sqlite3
import logging
logger = logging.getLogger(__name__)

class DatabaseInterface():
    def __init__(self, db_file: str):
        self.db_file = db_file

    def import_data_from_csv(self, csv_file: str, table_name: str):
        df = pd.read_csv(csv_file)
        df['id'] = df.index
        conn = sqlite3.connect(self.db_file)
        df.to_sql(table_name, con=conn, if_exists='replace', index=False)
        conn.close()
        logger.info(f"{df.shape[0]} rows stored into {table_name} in {self.db_file}")

    def get_ids(self, table_name: str) -> list[int]:
        try:
            conn = sqlite3.connect(self.db_file)
            df = pd.read_sql(f"SELECT id FROM {table_name}", con=conn)
            ids = df['id'].tolist()
            return ids
        except Exception as e:
            logger.error(f"Error retrieving ids from {table_name}: {e}")
            raise

    def get_texts(self, table_name: str, texts_col_name: str) -> list[str]:
        try:
            conn = sqlite3.connect(self.db_file)
            df = pd.read_sql(f"SELECT {texts_col_name} FROM {table_name}", con=conn)
            texts = df[f"{texts_col_name}"].tolist()
            return texts
        except Exception as e:
            logger.error(f"Error retrieving texts from {table_name}: {e}")
            raise

    def get_metadata(self, table_name: str, metadata_col_names: list[str]) -> list[dict]:
        try:
            conn = sqlite3.connect(self.db_file)
            df = pd.read_sql(f"SELECT {', '.join(metadata_col_names)} FROM {table_name}", con=conn)
            metadata = df.to_dict(orient='records')
            return metadata
        except Exception as e:
            logger.error(f"Error retrieving metadata from {table_name}: {e}")
            raise

    def get_rows_by_ids(self, ids: list[int], table_name: str) -> list[dict]:
        try:
            conn = sqlite3.connect(self.db_file)
            df = pd.read_sql(f"SELECT * FROM {table_name} WHERE id IN ({', '.join(map(str, ids))})", con=conn)
            return df.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error retrieving rows by ids from {table_name}: {e}")
            raise

