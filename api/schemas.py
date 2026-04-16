from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class ScriptureBase(BaseModel):
    volume_title: Optional[str]
    book_title: Optional[str]

class ScriptureRequest(ScriptureBase):
    query: str
    limit: int = Field(default=5, gt=0, description="Number of results to return")

class ScriptureResponse(ScriptureBase):
    id: int
    verse_title: str
    text: str