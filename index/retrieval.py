from scripture_storage import ScriptureStorage, Verse
from scripture_embedding import ScriptureEmbedder
from scripture_index import ScriptureIndex

class Retrieval:
    def __init__(self, FILE_PATH):
        # initialize storage
        self.storage = ScriptureStorage(FILE_PATH)
        texts = self.storage.get_all_texts()

        # initialize embedder
        self.embedder = ScriptureEmbedder()
        embeddings = self.embedder.embed_texts(texts)

        # inititalize index
        self.index = ScriptureIndex(embeddings)
        
    def query(self, query: str, k=5) -> list[Verse]:
        # embed query
        embedded_query = self.embedder.embed_single_text(query)

        # query index
        distances, indices = self.index.search(embedded_query, k=k)

        # pull verses
        verses = self.storage.get_verses(indices.flatten())

        return verses