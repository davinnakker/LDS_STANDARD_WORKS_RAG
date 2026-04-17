from retrieval.database import DatabaseInterface
from retrieval.vector_storage import VectorDB
from retrieval.embeddings import OpenAIEmbedder, Embedder

class IngestionService():
    def __init__(self, database: DatabaseInterface, embedder: Embedder, vector_db: VectorDB):
        self.database = database
        self.embedder = embedder
        self.vector_db = vector_db

    def retrieve_from_db(self, table_name: str, text_column: str, id_column: str, metadata_col_names: list[str]):
        self.ids = self.database.get_ids(table_name)
        self.texts = self.database.get_texts(table_name, text_column)
        self.metadata = self.database.get_metadata(table_name, metadata_col_names)
    
    def embed(self):
        self.vectors = self.embedder.encode_batch(self.texts)

    def store_in_vector_db(self, collection_name: str):
        self.vector_db.store_vectors(collection_name=collection_name, vectors=self.vectors, payloads=self.metadata, ids=self.ids)

class RetrievalService:
    def __init__(self, database: DatabaseInterface, embedder: Embedder, vector_db: VectorDB):
        self.database = database
        self.embedder = embedder
        self.vector_db = vector_db

    def retrieve(self, collection_name: str, table_name: str, query: str, top_k: int, filter: dict | None = None):
        query_vector = self.embedder.encode_single(query)
        ids = self.vector_db.search_vectors(collection_name=collection_name, query_vector=query_vector, top_k=top_k, filter=filter)
        rows = self.database.get_rows_by_ids(ids, table_name=table_name)
        return rows
        

    