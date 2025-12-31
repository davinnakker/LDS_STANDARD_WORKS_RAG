from .scripture_storage import ScriptureStorage, Verse
from .scripture_embedding import ScriptureEmbedder
from .scripture_index import ScriptureIndex
import numpy as np
import textwrap
import os

class Retrieval:
    def __init__(self, FILE_PATH, embeddings_cache_path="embeddings_cache.npy"):
        # initialize storage
        self.storage = ScriptureStorage(FILE_PATH)
        texts = self.storage.get_all_texts()

        # initialize embedder
        self.embedder = ScriptureEmbedder()

        # create embeddings from cache, or from model
        if os.path.exists(embeddings_cache_path):
            print(f"loading embeddings from {embeddings_cache_path}....")
            embeddings = np.load(embeddings_cache_path)
        else:
            embeddings = self.embedder.embed_texts(texts)
            np.save(embeddings_cache_path, embeddings)

        # inititalize index
        self.index = ScriptureIndex(embeddings)
        
    def query(self, query: str, k=5) -> list[Verse]:
        # embed query
        embedded_query = self.embedder.embed_single_text(query)

        # query index
        distances, indices = self.index.search(embedded_query, k=k)

        # pull verses
        self.verses = self.storage.get_verses(indices.flatten())

        return self.verses
    
    def display(self):
        print()
        for verse in self.verses:
            print(verse.citation)
            print(textwrap.fill(verse.text, width=40))
            print()