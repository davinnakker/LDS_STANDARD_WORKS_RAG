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
    def __init__(
        self, 
        client: OpenAI, 
        model_name: str = "text-embedding-3-small",
        batch_size: int = 500          # Safe default, you can increase up to 2048
    ):
        self.model_name = model_name
        self.client = client
        self.batch_size = batch_size   # Controls how many texts we send per API call

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text (kept for convenience)."""
        response = self.client.embeddings.create(
            input=text, 
            model=self.model_name
        )
        return response.data[0].embedding

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts with internal batching to avoid overloading the API."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        
        # Process texts in smaller batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch, 
                    model=self.model_name
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"Processed batch {i//self.batch_size + 1} "
                      f"({len(batch)} texts)")

            except RateLimitError as e:
                print(f"Rate limit hit. Waiting 60 seconds... Error: {e}")
                time.sleep(60)
                # Optional: retry the current batch once
                response = self.client.embeddings.create(input=batch, model=self.model_name)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            except APIError as e:
                print(f"API error on batch: {e}")
                time.sleep(10)  # Short backoff
                # You can add retry logic here if needed

        return all_embeddings
    
"""
class OpenAIEmbedder(Embedder):
    def __init__(self, client: OpenAI, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = client

    def encode_single(self, text):
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding
    
    def encode_batch(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in response.data]
"""