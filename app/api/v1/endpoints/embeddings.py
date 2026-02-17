from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.embedding import EmbeddingRequest, EmbeddingResponse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from app.core.config import settings
import os
import uuid

router = APIRouter()

@router.post("/", response_model=EmbeddingResponse, status_code=status.HTTP_201_CREATED)
async def create_embedding(
    request: EmbeddingRequest,
):
    """
    Embed raw text and store it in the vector database.
    
    - **text**: The raw text content to be embedded.
    - **metadata**: Optional JSON metadata (e.g. source, author, timestamp).
    - **namespace**: Not used for FAISS (local file storage).
    """
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
        
        # Create document
        doc_id = str(uuid.uuid4())
        document = Document(
            page_content=request.text,
            metadata={"id": doc_id, "source": "api_upload"}
        )
        
        vector_store_path = settings.VECTOR_STORE_PATH
        
        # Load or create FAISS index
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(
                vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents([document])
        else:
            vector_store = FAISS.from_documents([document], embeddings)
            
        # Save updated index locally
        vector_store.save_local(vector_store_path)

        # Force LangGraph service to reload the index
        try:
            from app.services.langgraph_rag import get_langgraph_service
            langgraph_service = get_langgraph_service()
            langgraph_service.reload_vector_store()
        except Exception as e:
            print(f"Warning: Failed to reload LangGraph vector store: {e}")

        return EmbeddingResponse(id=doc_id, status="success")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing embedding: {str(e)}"
        )
