"""
This file will take a list of vector embeddings and
the metadata needed and store it into a vector database
""" 

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import pandas as pd
from typing import List
import time

def make_points(df: pd.DataFrame, vectors):
    points = []
    for row in df.itertuples():
        point = PointStruct(
            id=row.id,
            vector=vectors[int(row.id)],
            payload={
                "id": row.id,
                "volume_title": row.volume_title,
                "book_title": row.book_title,
                "chapter": row.chapter_number,
                "verse": row.verse_number})
        points.append(point)
    return points


class VectorStore:
    def __init__(self, client: QdrantClient):
        self.client = client

    def create_collection(self, collection_name: str, dimension: int) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if collection_name in {c.name for c in collections}:
            print(f"Collection '{collection_name}' already exists.")
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance="Cosine")
        )
        print(f"Collection '{collection_name}' created successfully.")

    def upsert_vectors(
        self,
        collection_name: str,
        points: List[PointStruct],
        batch_size: int = 100,      # ← Lowered default (safer for large vectors)
        wait: bool = True,
        timeout: int = 120          # ← Higher timeout per batch (in seconds)
    ) -> None:
        """Upsert points in batches with retry logic and better error handling."""
        if not points:
            print("No points to upsert.")
            return

        batch: List[PointStruct] = []
        total_points = len(points)

        for i, point in enumerate(points, 1):
            batch.append(point)

            # When batch is full or it's the last point
            if len(batch) == batch_size or i == total_points:
                self._upsert_batch(
                    collection_name=collection_name,
                    batch=batch,
                    wait=wait,
                    timeout=timeout,
                    batch_number=(i // batch_size) + (1 if i % batch_size != 0 else 0),
                    total_batches=(total_points + batch_size - 1) // batch_size
                )
                batch.clear()

        print(f"✅ Successfully upserted all {total_points} points into '{collection_name}'.")

    def _upsert_batch(
        self,
        collection_name: str,
        batch: List[PointStruct],
        wait: bool,
        timeout: int,
        batch_number: int,
        total_batches: int
    ) -> None:
        """Internal helper with retry on timeout."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=wait,
                    timeout=timeout
                )
                print(f"✓ Batch {batch_number}/{total_batches} "
                      f"({len(batch)} points) uploaded successfully.")
                return

            except Exception as e:  # Catches WriteTimeout, ResponseHandlingException, etc.
                if attempt == max_retries - 1:
                    print(f"✗ Failed to upload batch {batch_number} after {max_retries} attempts.")
                    raise
                else:
                    wait_time = 10 * (attempt + 1)
                    print(f"⚠ Timeout on batch {batch_number} (attempt {attempt+1}). "
                          f"Retrying in {wait_time} seconds... Error: {type(e).__name__}")
                    time.sleep(wait_time)

                    # Optional: reduce batch size on retry for stubborn batches
                    if attempt >= 1:
                        print("   Reducing batch size for next retry.")
                        # You can handle this by slicing in the main loop if needed

"""   
class VectorStore:
    def __init__(self, client: QdrantClient):
        self.client = client
    
    def create_collection(self, collection_name: str, dimension: int):
        if collection_name not in [collection.name for collection in self.client.get_collections().collections]:
            self.client.create_collection(collection_name, VectorParams(size=dimension, distance='Cosine'))

    def upsert_vectors(self, collection_name: str, points: list[PointStruct], batch_size: int = 150):
        batch = []
        for i, point in enumerate(points):
            batch.append(point)
            if len(batch) == batch_size:
                self.client.upsert(collection_name, points=batch, timeout=200)
                print(f"{i} / {len(points)}")
                batch.clear()
            

        if batch:
            self.client.upsert(collection_name, points=batch, timeout=200)
"""
