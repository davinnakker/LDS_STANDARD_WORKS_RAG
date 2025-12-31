from fastapi import FastAPI
import uvicorn
from index.retrieval import Retrieval

app = FastAPI(title="Scripture Retriever",
              description="Semantically search the lds standard works")

retriever = None

@app.on_event("startup")
def startup():
    # set inputs for retrieval
    SCRIPTURE_FILE = "index/data/lds-scriptures.csv"
    CACHE_FILE = "index/embeddings_cache.npy"

    # initiate retrieval class
    global retriever
    retriever = Retrieval(SCRIPTURE_FILE, embeddings_cache_path=CACHE_FILE)

@app.get("/search")
def query(query: str, k: int) -> list[dict]:
    verses = retriever.query(query, k=k)

    results = []
    for verse in verses:
        verse_dict = {"citation": verse.citation,
                      "text": verse.text}
        
        results.append(verse_dict)
    
    return results


# to run
if __name__ == "__main__":
    uvicorn.run("main:app",
                host="127.0.0.1",
                port=8000,
                reload=True)
