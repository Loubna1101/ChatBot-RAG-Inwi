from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse, ResetRequest, ResetResponse, HealthResponse
from agent.agent import agent
from rag.vectorstore import get_collection_stats

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        session_id = req.session_id or agent.create_session()
        result = agent.chat(req.message, session_id)
        return ChatResponse(answer=result["answer"], sources=result["sources"], session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-conversation", response_model=ResetResponse)
def reset(req: ResetRequest):
    result = agent.reset(req.session_id)
    return ResetResponse(**result)

@router.get("/health", response_model=HealthResponse)
def health():
    try:
        stats = get_collection_stats()
        return HealthResponse(status="ok", collection_stats=stats)
    except Exception as e:
        return HealthResponse(status="error", collection_stats={"error": str(e)})