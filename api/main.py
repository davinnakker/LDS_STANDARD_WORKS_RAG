from fastapi import FastAPI
from .schemas import *
from .services import get_retrieval_service

app = FastAPI()

retrieval = get_retrieval_service()

@app.post("/scriptures/search")
def search(request: ScriptureRequest) -> list[ScriptureResponse]:
    if request.volume_title and request.book_title:
        filter = {"volume_title": request.volume_title, "book_title": request.book_title}
    elif request.volume_title:
        filter = {"volume_title": request.volume_title}
    else:
        filter = None

    print(f"Filter first: {filter}")

    results = retrieval.retrieve(query=request.query, db_table_name="standard_works", vb_collection_name="standard_works_openai", top_k=request.limit, filter=filter)

    responses = [ScriptureResponse(id=result['id'], 
                                   volume_title=result['volume_title'], 
                                   book_title=result['book_title'], 
                                   verse_title=result['verse_title'], 
                                   text=result['scripture_text']) for result in results]
    
    return responses

@app.post("/general_conference/search")
def search(request):
    pass

