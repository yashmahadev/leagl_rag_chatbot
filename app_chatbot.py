# app_chatbot_advanced_refined.py
import streamlit as st
import os
import time
from hybrid_retriever_fixed import retrieve
import ollama

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Indian Legal AI Assistant", page_icon="⚖️", layout="centered")

# ------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "show_citations" not in st.session_state:
    st.session_state.show_citations = True

# ------------------------------------------------
# STYLE SETUP
# ------------------------------------------------
bg_color = "#0f172a" if st.session_state.dark_mode else "#F9FAFB"
text_color = "#F9FAFB" if st.session_state.dark_mode else "#111827"
bot_bg = "#1E293B" if st.session_state.dark_mode else "#E5E7EB"
user_bg = "#2563EB"
input_bg = "#1E293B" if st.session_state.dark_mode else "#FFFFFF"
border_color = "#3B82F6"

st.markdown(f"""
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: {bg_color};
            color: {text_color};
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
            line-height: 1.5;
            font-size: 16px;
            word-wrap: break-word;
        }}
        .user-msg {{
            background-color: {user_bg};
            color: white;
            text-align: right;
            border-bottom-right-radius: 4px;
        }}
        .bot-msg {{
            background-color: {bot_bg};
            color: {text_color};
            border-bottom-left-radius: 4px;
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
            background-color: {input_bg};
            border: 2px solid {border_color};
            border-radius: 25px;
            padding: 8px 20px;
        }}
        .stTextInput > div > div > input {{
            color: {text_color} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown(f"""
        <div style="text-align:center;">
            <h1>⚖️ Indian Legal AI Assistant</h1>
            <p style="color:{'#9CA3AF' if st.session_state.dark_mode else '#4B5563'};">
                Ask about IPC, CrPC, or NIA — get precise, section-wise answers with citations.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.show_citations = st.toggle("📜 Show Citations", value=st.session_state.show_citations)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------
# RESPONSE GENERATION
# ------------------------------------------------
def generate_stream_response(query, context, show_citations=True):
    """Stream tokens like ChatGPT."""
    citation_instruction = (
        "Include section numbers and act names clearly."
        if show_citations else
        "Avoid legal citations, summarize in plain English."
    )
    prompt = f"""
You are a senior Indian legal expert assistant.
You must answer precisely using the context below when relevant.

Context:
{context}

Question:
{query}

Guidelines:
- {citation_instruction}
- Ensure accuracy >99%. Use context for legal facts.
- If unrelated, respond as a knowledgeable AI assistant conversationally.
    """

    response_text = ""
    stream = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}], stream=True)
    message_placeholder = st.empty()

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            response_text += content
            message_placeholder.markdown(f'<div class="chat-bubble bot-msg">{response_text}</div>', unsafe_allow_html=True)
            time.sleep(0.03)

    return response_text.strip()

# ------------------------------------------------
# CHAT HISTORY DISPLAY
# ------------------------------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------
# CHAT INPUT
# ------------------------------------------------
user_input = st.chat_input("Type your legal question here...", disabled=st.session_state.is_generating)

if user_input and not st.session_state.is_generating:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.is_generating = True
    st.rerun()

# ------------------------------------------------
# HANDLE RESPONSE
# ------------------------------------------------
if st.session_state.is_generating:
    with st.spinner("💭 Thinking... analyzing your question..."):
        query = st.session_state.messages[-1]["content"]
        results = retrieve(query, top_k=3)

        context = "\n\n".join([
            f"{r['act']} Section {r['section']}: {r['title']}\n{r['text']}"
            for r in results
        ]) if results else ""

        answer = generate_stream_response(query, context, st.session_state.show_citations)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.is_generating = False
        st.rerun()
