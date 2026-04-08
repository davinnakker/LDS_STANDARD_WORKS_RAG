from .database import DataBase
from .embeddings import OpenAIEmbedder, Embedder
from .vector_storage import VectorStorage
import pandas as pd

class IngestionService():
    def __init__(self, database: DataBase, embedder: Embedder, vector_db: VectorStorage):
        self.database = database
        self.embedder = embedder
        self.vector_db = vector_db

        self.df = None
        self.texts = []
        self.embeddings = []

    def save_to_database(self, csv_file: str, table_name: str, text_col_name: str):
        self.database.add_table_via_csv(csv_file=csv_file, table_name=table_name)
        self.texts, self.df = self.database.get_texts_and_metadata(table_name, text_col_name)

    def save_to_database(self, df: pd.DataFrame, table_name: str, text_col_name: str):
        self.database.add_table_via_df(table_name, df)
        self.texts, self.df = self.database.get_texts_and_metadata(table_name, text_col_name)

    def make_embeddings(self):
        self.embeddings = self.embedder.encode_batch(self.texts)
    
    def store_vectors(self, collection_name):
        self.vector_db.upload_collection(self.df, self.embeddings, collection_name)

class RetrievalService:
    def __init__(self, database: DataBase, embedder: Embedder, vector_db: VectorStorage):
        self.database = database
        self.embedder = embedder
        self.vector_db = vector_db

    def retrieve(self):
        pass

    