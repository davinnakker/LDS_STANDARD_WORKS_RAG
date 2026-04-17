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

class IngestRequest(BaseModel):
    table_name: str
    text_column: str
    id_column: str
    metadata_col_names: list[str]
    collection_name: str

    model_config = ConfigDict(from_attributes=True)