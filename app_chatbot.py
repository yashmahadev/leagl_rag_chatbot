# app_chatbot_advanced_refined.py
import streamlit as st
import time
import ollama
from hybrid_retriever import retrieve

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="⚖️ Indian Legal AI Assistant", layout="centered")

# ------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------
for key, default in {
    "messages": [],
    "is_generating": False,
    "dark_mode": False,
    "show_citations": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------
# COLOR THEMES
# ------------------------------------------------
def get_theme():
    return {
        "bg": "#0f172a" if st.session_state.dark_mode else "#F9FAFB",
        "text": "#F9FAFB" if st.session_state.dark_mode else "#111827",
        "bot_bg": "#1E293B" if st.session_state.dark_mode else "#E5E7EB",
        "user_bg": "#2563EB",
        "input_bg": "#1E293B" if st.session_state.dark_mode else "#FFFFFF",
        "border": "#3B82F6",
    }

theme = get_theme()

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown(f"""
    <style>
        body {{
            background-color: {theme['bg']};
            color: {theme['text']};
            font-family: 'Inter', sans-serif;
        }}
        .chat-container {{
            max-width: 700px;
            margin: auto;
            padding-bottom: 100px;
        }}
        .chat-bubble {{
            border-radius: 16px;
            padding: 12px 16px;
            margin-bottom: 10px;
            line-height: 1.6;
            font-size: 16px;
        }}
        .user-msg {{
            background-color: {theme['user_bg']};
            color: white;
            text-align: right;
        }}
        .bot-msg {{
            background-color: {theme['bot_bg']};
            color: {theme['text']};
        }}
        .thinking {{
            font-style: italic;
            color: #9CA3AF;
            animation: blink 1.2s infinite;
        }}
        @keyframes blink {{
            50% {{ opacity: 0.4; }}
        }}
        .input-box {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 700px;
            background-color: {theme['input_bg']};
            border: 2px solid {theme['border']};
            border-radius: 25px;
            padding: 8px 20px;
        }}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown(f"""
<div style="text-align:center;">
    <h1>⚖️ Indian Legal AI Assistant</h1>
    <p style="color:{'#9CA3AF' if st.session_state.dark_mode else '#4B5563'};">
        Get section-wise, accurate legal insights from IPC, CrPC, and NIA Acts.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", st.session_state.dark_mode)
    st.session_state.show_citations = st.toggle("📜 Show Citations", st.session_state.show_citations)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------
# RESPONSE GENERATION
# ------------------------------------------------
def generate_stream_response(query, context, show_citations=True):
    """Stream ChatGPT-like response with structured legal reasoning."""
    citation_instruction = (
        "Include exact section numbers, act names, and cite them clearly."
        if show_citations else
        "Explain without mentioning act or section numbers."
    )

    system_prompt = """
    You are a Supreme Court-level Indian Legal Assistant.
    You must answer precisely and factually based only on provided legal context.
    Always cite relevant sections and acts (IPC, CrPC, or NIA) when context allows.
    """

    prompt = f"""
    Context:
    {context}

    User Question:
    {query}

    Guidelines:
    - {citation_instruction}
    - Ensure 99%+ factual accuracy.
    - Never hallucinate or invent sections.
    - Use this structure:
        ⚖️ **Act & Section:**
        📘 **Legal Explanation:**
        🧩 **Example / Practical Insight:**
    """

    response_text = ""
    placeholder = st.empty()

    # Stream tokens from Ollama
    stream = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    for chunk in stream:
        token = chunk.get("message", {}).get("content", "")
        if token:
            response_text += token
            placeholder.markdown(f'<div class="chat-bubble bot-msg">{response_text}</div>', unsafe_allow_html=True)
            time.sleep(0.02)

    return response_text.strip()

# ------------------------------------------------
# CHAT DISPLAY
# ------------------------------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    bubble_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    st.markdown(f'<div class="chat-bubble {bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------
# INPUT FIELD
# ------------------------------------------------
user_input = st.chat_input("Ask your legal question here...", disabled=st.session_state.is_generating)

if user_input and not st.session_state.is_generating:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.is_generating = True
    st.rerun()

# ------------------------------------------------
# HANDLE GENERATION
# ------------------------------------------------
if st.session_state.is_generating:
    with st.spinner("💭 Thinking... Analyzing your question..."):
        query = st.session_state.messages[-1]["content"]
        retrieved_docs = retrieve(query, top_k=3)

        context = "\n\n".join([
            f"{r.get('act', '')} Section {r.get('section', '')}: {r.get('title', '')}\n{r['text']}"
            for r in retrieved_docs
        ]) if retrieved_docs else "No matching sections found."

        answer = generate_stream_response(query, context, st.session_state.show_citations)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.is_generating = False
        st.rerun()
