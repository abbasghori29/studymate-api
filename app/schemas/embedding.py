from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="The raw text to embed")

class EmbeddingResponse(BaseModel):
    id: str = Field(..., description="ID of the stored embedding")
    status: str = Field(default="success", description="Status of the operation")
