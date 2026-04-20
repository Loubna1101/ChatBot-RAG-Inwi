import uuid
from rag.pipeline import run_rag_pipeline
from rag import memory

class InwiAgent:
    def create_session(self):
        return str(uuid.uuid4())

    def chat(self, message: str, session_id: str) -> dict:
        if not message or not message.strip():
            return {"answer": "Veuillez poser une question.", "sources": [], "session_id": session_id}
        result = run_rag_pipeline(message.strip(), session_id)
        return {"answer": result["answer"], "sources": result["sources"], "session_id": session_id}

    def reset(self, session_id: str):
        memory.reset_session(session_id)
        return {"status": "reset", "session_id": session_id}

agent = InwiAgent()