import streamlit as st
import requests
import uuid
import base64
import os

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="inwi Assistant", page_icon="inwi.png", layout="centered")

def get_logo_base64():
    logo_path = "inwi.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()

st.markdown("""
<style>
.stApp { background-color: #0a0a0a !important; }
[data-testid="stSidebar"] { background-color: #111111 !important; border-right: 1px solid #6B2D8B !important; }
html, body, .stApp, .stMarkdown, p, span, label, div { color: #ffffff !important; }
h1, h2, h3 { color: #C084FC !important; }
.inwi-header { background: linear-gradient(135deg, #1a0a2e 0%, #2d1a4a 50%, #1a0a2e 100%); border: 1px solid #6B2D8B; border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 24px; box-shadow: 0 0 30px rgba(107, 45, 139, 0.3); }
.inwi-header h1 { color: #C084FC !important; font-size: 2rem; margin: 0; letter-spacing: 2px; }
.inwi-header p { color: #a78bca !important; margin: 8px 0 0 0; font-size: 0.9rem; }
.inwi-header img { height: 50px; margin-bottom: 10px; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background-color: #1e0a3c !important; border: 1px solid #6B2D8B !important; border-radius: 12px !important; padding: 12px !important; margin: 8px 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background-color: #150d1f !important; border: 1px solid #3d1f5c !important; border-radius: 12px !important; padding: 12px !important; margin: 8px 0 !important; }
[data-testid="stChatInput"] { background-color: #1a0a2e !important; border: 2px solid #6B2D8B !important; border-radius: 12px !important; }
[data-testid="stChatInput"] textarea { background-color: #1a0a2e !important; color: #ffffff !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #9b72c0 !important; }
.stButton > button { background: linear-gradient(135deg, #6B2D8B, #9333EA) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; transition: all 0.3s ease !important; }
.stButton > button:hover { background: linear-gradient(135deg, #9333EA, #C084FC) !important; box-shadow: 0 4px 15px rgba(147, 51, 234, 0.4) !important; transform: translateY(-1px) !important; }
.stSpinner > div { border-top-color: #9333EA !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #6B2D8B; border-radius: 3px; }
.stCaption { color: #9b72c0 !important; }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

def call_api(message):
    try:
        r = requests.post(f"{API_URL}/chat",
            json={"message": message, "session_id": st.session_state.session_id},
            timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"answer": f"❌ Erreur : {str(e)}", "sources": [], "session_id": st.session_state.session_id}

if logo_b64:
    st.markdown(f"""
    <div class='inwi-header'>
        <img src="data:image/png;base64,{logo_b64}" alt="inwi logo">
        <h1>inwi Assistant</h1>
        <p>مرحباً بك • Bienvenue • Posez vos questions sur les services inwi</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='inwi-header'>
        <h1>inwi Assistant</h1>
        <p>مرحباً بك • Bienvenue • Posez vos questions sur les services inwi</p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Options")
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        try:
            requests.post(f"{API_URL}/reset-conversation",
                          json={"session_id": st.session_state.session_id})
        except:
            pass
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("### 📞 Contacts inwi")
    st.markdown("📱 **Service client** : 120")
    st.markdown("🌐 **Site web** : inwi.ma")
    st.markdown("---")
    st.caption(f"Session : `{st.session_state.session_id[:8]}...`")

for i, msg in enumerate(st.session_state.messages):
    icon = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.markdown("#### 💡 Questions fréquentes")
    cols = st.columns(2)
    examples = [
        "Quels sont les forfaits Illimités Inwi ?",
        "Comment consulter mon solde ?",
        "كيفاش نشحن رصيدي ؟",
        "Comment activer l'eSIM ?"
    ]
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, use_container_width=True):
                st.session_state._eq = ex
                st.rerun()

if hasattr(st.session_state, "_eq"):
    q = st.session_state._eq
    del st.session_state._eq
    st.session_state.messages.append({"role": "user", "content": q})
    result = call_api(q)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.rerun()

if prompt := st.chat_input("💬 Posez votre question en français ou بالعربية..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("⏳ Réflexion..."):
            result = call_api(prompt)
        st.markdown(result["answer"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})