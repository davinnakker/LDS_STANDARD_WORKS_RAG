from fastapi import FastAPI
from index.retrieval import Retrieval

app = FastAPI(title="Scripture Retriever",
              description="Semantically search the lds standard works")

@app.on_event("start-up")
def startup():
    # set inputs for retrieval
    SCRIPTURE_FILE = "index\data\lds-scriptures.csv"
    CACHE_FILE = "index\embeddings_cache.npy"

    # initiate retrieval class
    global retrieval
    retrieval = Retrieval(SCRIPTURE_FILE, embeddings_cache_path=CACHE_FILE)

