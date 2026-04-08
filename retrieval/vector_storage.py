"""
This file will take a list of vector embeddings and
the metadata needed and store it into a vector database
""" 

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import pandas as pd
from typing import List
import time
from numpy import ndarray

  
class VectorStorage:
    def __init__(self, client: QdrantClient):
        self.client = client
    
    def upload_collection(self, df: pd.DataFrame, vectors: ndarray, collection_name: str):
        # create collection
        dimension = len(vectors[0])
        if collection_name not in self.client.get_collections().collections:
            self.client.recreate_collection(collection_name, VectorParams(size=dimension, distance=Distance.COSINE))
        
        # preprocess and upload
        ids = df['id'].tolist()
        payloads = df[['volume_title', 'book_title', 'id']].to_dict(orient="records")

        assert len(ids) == len(payloads) == len(vectors)

        self.client.upload_collection(collection_name=collection_name,
                                      vectors=vectors,
                                      payload=payloads,
                                      ids=ids,
                                      batch_size=100,
                                      parallel=2)
        
        

        