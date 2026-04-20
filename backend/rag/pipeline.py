import os
import re
from groq import Groq
from sentence_transformers import SentenceTransformer
from rag import vectorstore, memory
from dotenv import load_dotenv

load_dotenv()

_embedding_model = None
_llm_client = None

def get_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    return _embedding_model

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _llm_client

SYSTEM_PROMPT = """Tu es l'assistant virtuel officiel d'inwi, opérateur télécom marocain.
Tu réponds aux questions sur les services inwi en t'appuyant sur le contexte fourni.

RÈGLES :
1. Toutes les questions posées concernent inwi par défaut.
2. Utilise le contexte fourni pour répondre. Si le contexte contient des informations même partiellement liées, utilise-les pour construire ta réponse.
3. Si le contexte ne contient AUCUNE information liée à la question, dis : "Je ne dispose pas de cette information. Contactez le service client inwi au 120 ou via inwi.ma"
4. N'invente JAMAIS de tarifs, horaires ou procédures qui ne sont pas dans le contexte.
5. Sois poli, concis, professionnel.
6. Réponds dans la langue de l'utilisateur (français, arabe ou darija).
7. Si la question est clairement hors sujet (météo, politique...), dis : "Je suis uniquement formé pour répondre aux questions sur les services inwi."
8. Ne mentionne jamais de concurrents.
9. Si l'utilisateur pose une question de suivi, utilise l'historique pour comprendre le contexte.

CONTEXTE (documents inwi) :
{context}

HISTORIQUE DE LA CONVERSATION :
{history}
"""

BLOCKED_PATTERNS = [
    (r"\b(météo|temps qu'il fait|température|climat)\b", "Je suis uniquement formé pour répondre aux questions sur les services inwi."),
    (r"\b(président|roi|gouvernement|politique)\b", "Je suis uniquement formé pour répondre aux questions sur les services inwi."),
    (r"\b(pirater|hacker|voler un compte|frauder)\b", "Je ne peux pas aider avec des demandes non éthiques."),
    (r"\b(maroc telecom|orange|iam)\b", "Je suis l'assistant inwi et ne compare pas avec d'autres opérateurs."),
]

def check_guardrails(query):
    for pattern, response in BLOCKED_PATTERNS:
        if re.search(pattern, query.lower(), re.IGNORECASE):
            return response
    return None

def rewrite_query_with_history(question: str, session_id: str) -> str:
    history = memory.get_history(session_id)
    if not history:
        return question

    last_messages = history[-4:]

    history_text = ""
    for msg in last_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content'][:200]}\n"

    client = get_llm_client()
    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": """Reformule la question de l'utilisateur en une question autonome et complète, 
en intégrant le contexte de la conversation précédente.
Réponds UNIQUEMENT avec la question reformulée, rien d'autre.
Si la question est déjà autonome, retourne-la telle quelle."""},
            {"role": "user", "content": f"Historique:\n{history_text}\nNouvelle question: {question}\n\nQuestion reformulée:"}
        ],
        temperature=0,
        max_tokens=150
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"  🔄 Question originale:  {question}")
    print(f"  🔄 Question reformulée: {rewritten}")
    return rewritten

def run_rag_pipeline(question: str, session_id: str) -> dict:
    guardrail_response = check_guardrails(question)
    if guardrail_response:
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", guardrail_response)
        return {"answer": guardrail_response, "sources": []}

    search_query = rewrite_query_with_history(question, session_id)

    model = get_model()
    query_embedding = model.encode(
        f"query: {search_query}",
        normalize_embeddings=True
    ).tolist()

    retrieved_docs = vectorstore.search(query_embedding, top_k=5)

    print(f"\n🔍 Question: {question}")
    print(f"📄 Documents récupérés: {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs):
        print(f"   #{i+1} score={doc['score']} | {doc['metadata'].get('source','')} | {doc['text'][:80]}...")

    if retrieved_docs:
        context = "\n\n---\n\n".join([
            f"[Source: {doc['metadata'].get('source', 'inwi')}]\n{doc['text']}"
            for doc in retrieved_docs
        ])
    else:
        context = "Aucun document pertinent trouvé."

    history_text = memory.get_history_as_text(session_id) or "Pas d'historique."

    client = get_llm_client()
    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context, history=history_text)},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    memory.add_message(session_id, "user", question)
    memory.add_message(session_id, "assistant", answer)

    sources = [
        {
            "source": d["metadata"].get("source", ""),
            "section": d["metadata"].get("h1", "") or d["metadata"].get("h2", ""),
            "score": d["score"]
        }
        for d in retrieved_docs
    ]
    return {"answer": answer, "sources": sources}