from retrieval.orchestration import RetrievalService, IngestionService
from retrieval.database import DatabaseInterface
from retrieval.embeddings import OpenAIEmbedder
from retrieval.vector_storage import VectorDB
from openai import OpenAI
from qdrant_client import QdrantClient
from settings.config import Config
settings = Config()

# configure services
database = DatabaseInterface("data/scriptures.db")

openai_client = OpenAI(api_key=settings.open_api_key)
embedder = OpenAIEmbedder(client=openai_client)

qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=1000)
vector_db = VectorDB(qdrant_client)


# get service functions
def get_retrieval_service():
    return RetrievalService(database=database, embedder=embedder, vector_db=vector_db)

def get_ingestion_service():
    return IngestionService(database=database, embedder=embedder, vector_db=vector_db)







