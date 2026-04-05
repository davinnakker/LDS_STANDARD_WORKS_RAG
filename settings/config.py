from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.open_api_key = os.getenv("OPENAI_TOKEN")
        