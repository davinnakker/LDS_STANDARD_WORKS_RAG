from fastapi import FastAPI
import uvicorn
from semantic.retrieval import Retrieval
from .helper_functions import get_csv


app = FastAPI(title="Scripture Retriever",
              description="Semantically search the lds standard works")

retrievers = {}

for book in [
    "All",
    "Old Testament",
    "New Testament",
    "Book of Mormon",
    "Doctrine and Covenants",
    "Pearl of Great Price"
]:
    csv, embeddings = get_csv(book)
    retrievers[book] = Retrieval(csv, embeddings)


@app.get("/search")
def query(query: str, k: int, book: str = 'All') -> list[dict]:

    retriever = retrievers[book]
    
    verses = retriever.query(query, k)

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
