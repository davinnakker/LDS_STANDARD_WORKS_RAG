"""
This file contains different embedder classes that can be used to take 
texts and turnthem into high dimensional vectors that represent semantic meaning
"""

from typing import List
from openai import OpenAI, RateLimitError, APIError
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
import time

class Embedder(ABC):

    @abstractmethod
    def encode_single(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        pass

class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print('loading model...')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def encode_single(self, text: str) -> list[float]:
        vector = self.model.encode(text)
        return vector
    
    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        print('embedding texts...')
        vectors = self.model.encode(texts, show_progress_bar=True)
        return vectors
    

class OpenAIEmbedder(Embedder):
    def __init__(self, client: OpenAI, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = client

    def encode_single(self, text):
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding
    
    def encode_batch(self, texts, batch_size: int = 300):
        all_embeddings = []
        total_texts = len(texts)
        print(f"Total texts to embed: {total_texts}")

        for i in range(0, total_texts, batch_size):
            # Slice batch safely
            batch = texts[i:i + batch_size]
            print(f"Processing batch: texts {i} through {i + len(batch) - 1}")

            # Call OpenAI embeddings API
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings = [item.embedding for item in response.data]

            # Append embeddings to the flat list
            all_embeddings.extend(embeddings)

            print(f"Embedded {len(all_embeddings)} of {total_texts} texts so far")

        return all_embeddings
