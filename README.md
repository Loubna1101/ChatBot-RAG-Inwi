# ChatBot RAG inwi

Agent conversationnel intelligent pour le support client inwi, basé sur RAG (Retrieval-Augmented Generation).

## Stack technique

- **LLM** : Llama 3.3 70B (via Groq)
- **Embeddings** : intfloat/multilingual-e5-base
- **Base vectorielle** : ChromaDB
- **Backend** : FastAPI
- **Frontend** : Streamlit
- **Déploiement** : Docker

## Lancement rapide

### Sans Docker

```bash
# Indexation
cd backend
pip install -r requirements.txt
python index_all.py

# Backend
uvicorn main:app --reload --port 8000

# Frontend (nouveau terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
