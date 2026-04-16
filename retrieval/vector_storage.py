"""
Core functionality from the a vector database
- create a collection if not already there
- replace a collection if already there
- upload a collection of vectors
- index certain payloads
- search a collection of vectors
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType
from typing import List
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class VectorStoring():

    def __init__(self, 
                 qdrant_client: QdrantClient,
                 collection_name: str,
                 vectors: List[List[float]],
                 distance: Distance = Distance.COSINE):
        
        self.qdrant_client = qdrant_client
        self.dimension = len(vectors[0]) if vectors else 0
        self.distance = distance
        self.collection_name = collection_name

    def create_collection(self):
        if self.collection_name in self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
        try:
            response = self.qdrant_client.create_collection(collection_name=self.collection_name, vectors_config=VectorParams(size=self.dimension, distance=self.distance))
            if response:
                logger.info(f"Collection '{self.collection_name}' created successfully with vector size {self.dimension} and distance '{self.distance}'.")
                return response
            else:
                logger.warning(f"Failed to create collection '{self.collection_name}'.")
                return response
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def upload_collection(self, vectors: List[List[float]], payloads: List[dict], ids: List[int]):
        if self.qdrant_client.collection_exists(self.collection_name):
            try:
                response = self.qdrant_client.upload_collection(collection_name=self.collection_name,
                                                    vectors=vectors,
                                                    payload=payloads,
                                                    ids=ids,
                                                    batch_size=100,
                                                    parallel=2,
                                                    timeout=1000)
                if response:
                    logger.info(f"Uploaded {len(ids)} vectors to collection '{self.collection_name}'.")
                    return response
                else:
                    logger.warning(f"Failed to upload vectors to collection '{self.collection_name}'.")
                    return response
            except Exception as e:
                logger.error(f"Error uploading collection: {e}")
                raise
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist. Please create it before uploading.")
        
    def index_payload(self, field_name: str):
        try:
            response = self.qdrant_client.create_payload_index(collection_name=self.collection_name, field_name=field_name, field_schema=PayloadSchemaType.KEYWORD)
            if response:
                logger.info(f"Payload index created successfully for field '{field_name}' in collection '{self.collection_name}'.")
                return response
            else:
                logger.warning(f"Failed to create payload index for field '{field_name}' in collection '{self.collection_name}'.")
                return response
        except Exception as e:
            logger.error(f"Error creating payload index: {e}")
            raise

class VectorSearch():
    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    def search_collection(self, query_vector: List[float], top_k: int, filter: dict | None = None):
        if filter is not None:
            filter = self.__make_filter(filter)
            response = self.qdrant_client.query_points(collection_name=self.collection_name, query=query_vector, limit=top_k, query_filter=filter)
        else:
            response = self.qdrant_client.query_points(collection_name=self.collection_name, query=query_vector, limit=top_k)
        logger.info(f"Search completed with response: {response}")
        ids = [point.id for point in response.points]
        return ids
    
    def __make_filter(metadata: dict):
            filter = Filter(must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in metadata.items()])
            return filter
    
class VectorDB():
    def __init__(self, qdrant_client: QdrantClient):
        self.vector_storing = VectorStoring(qdrant_client)
        self.vector_search = VectorSearch(qdrant_client)

    def store_vectors(self, collection_name: str, vectors: List[List[float]], payloads: List[dict], ids: List[int]):
        self.vector_storing.create_collection(collection_name)
        self.vector_storing.upload_collection(collection_name, vectors, payloads, ids)
        index_fields = [key for key in payloads[0].keys() if key != 'id']  # Exclude 'text' from indexing
        if index_fields:
            for field in index_fields:
                self.vector_storing.index_payload(collection_name, field)
        logger.info(f"Vectors stored successfully in collection '{collection_name}' with {len(ids)} entries.")

    def search_vectors(self, collection_name: str, query_vector: List[float], top_k: int, filter: dict | None = None):
        return self.vector_search.search_collection(collection_name, query_vector, top_k, filter)

    












