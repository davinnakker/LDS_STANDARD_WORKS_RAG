from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class ScriptureBase(BaseModel):
    volume: Optional[str]
    book: Optional[str]
    verse: Optional[int]

class ScriptureRequest(ScriptureBase):
    query: str

class ScriptureResponse(ScriptureBase):
    id: int
    verse_title: str
    text: str

    