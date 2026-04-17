from fastapi import FastAPI
from .schemas import *
from .services import get_retrieval_service, get_ingestion_service

app = FastAPI()

retrieval = get_retrieval_service()

@app.post("/ingest")
def ingest(request: IngestRequest):
    ingestion = get_ingestion_service()
    ingestion.retrieve_from_db(table_name=request.table_name, text_column=request.text_column, id_column=request.id_column, metadata_col_names=request.metadata_col_names)
    ingestion.embed()
    ingestion.store_in_vector_db(collection_name=request.collection_name)
    return {"message": f"Data from {request.table_name} ingested into {request.collection_name} vector database."}


@app.post("/scriptures/search")
def search(request: ScriptureRequest) -> list[ScriptureResponse]:
    if request.volume_title and request.book_title:
        filter = {"volume_title": request.volume_title, "book_title": request.book_title}
    elif request.volume_title:
        filter = {"volume_title": request.volume_title}
    else:
        filter = None

    print(f"Filter first: {filter}")

    results = retrieval.retrieve(collection_name="standard_works_openai", table_name="standard_works", query=request.query, top_k=request.limit, filter=filter)

    responses = [ScriptureResponse(id=result['id'], 
                                   volume_title=result['volume_title'], 
                                   book_title=result['book_title'], 
                                   verse_title=result['verse_title'], 
                                   text=result['scripture_text']) for result in results]
    
    return responses

@app.post("/general_conference/search")
def search(request):
    pass

