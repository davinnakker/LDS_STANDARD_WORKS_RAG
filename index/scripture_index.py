import numpy as np
from faiss import IndexFlatL2

class ScriptureIndex:
    def __init__(self, embeddings: np.ndarray[np.ndarray]):
        self.embeddings = embeddings
        self.embedding_dimensions = embeddings.shape[1]
        print("creating index....")
        self.index = IndexFlatL2(self.embedding_dimensions)
        self.index.add(embeddings)
        print("index created!")
        print(f"Vectors: {self.index.ntotal}\nDimensions: {self.index.d}")

    def search(self, vector, k=2):
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        distances, indices = self.index.search(vector, k)

        return distances, indices