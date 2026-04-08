from fastapi import FastAPI
from schemas import *

app = FastAPI()



@app.post("/scriptures/search")
def search(request: ScriptureRequest) -> ScriptureResponse:
    pass