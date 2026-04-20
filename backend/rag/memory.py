from collections import defaultdict

_sessions = defaultdict(list)
MAX_HISTORY = 5

def add_message(session_id: str, role: str, content: str):
    _sessions[session_id].append({"role": role, "content": content})
    max_messages = MAX_HISTORY * 2
    if len(_sessions[session_id]) > max_messages:
        _sessions[session_id] = _sessions[session_id][-max_messages:]

def get_history(session_id: str):
    return _sessions.get(session_id, [])

def reset_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]

def get_history_as_text(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return ""
    formatted = []
    for msg in history:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)
