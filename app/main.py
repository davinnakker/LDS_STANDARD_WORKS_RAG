from fastapi import FastAPI
import uvicorn
from scripture.retrieval import Retrieval

app = FastAPI(title="Scripture Retriever",
              description="Semantically search the lds standard works")

retriever = None

@app.on_event("startup")
def startup():
    # set inputs for retrieval (paths relative to project root)
    SCRIPTURE_FILE = "data/lds-scriptures.csv"
    CACHE_FILE = "data/embeddings_cache.npy"

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


# Run from project root: uvicorn app.main:app --reload
if __name__ == "__main__":
    uvicorn.run("app.main:app",
                host="127.0.0.1",
                port=8000,
                reload=True)
