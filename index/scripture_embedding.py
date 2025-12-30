from sentence_transformers import SentenceTransformer

class ScriptureEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print('loading model....')
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embeddings = None
        print(f"Model: {self.model_name}\nhas been loaded successfully!")

    def embed_single_text(self, text: str):
        embedding = self.model.encode([text], convert_to_numpy=True)

        return embedding
    
    def embed_texts(self, texts: list[str]):
        print("embedding texts....")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print("embeddings complete!")
        print(f"Embedding Vectors: {self.embeddings.shape[0]}")
        print(f"Dimensions: {self.embeddings.shape[1]}")

        return self.embeddings