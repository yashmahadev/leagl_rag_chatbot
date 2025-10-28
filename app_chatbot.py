# app_chatbot.py
import streamlit as st
import os
import time
import ollama
from hybrid_retriever import retrieve

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Indian Legal AI Assistant",
    page_icon="⚖️",
    layout="centered"
)

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
# DYNAMIC STYLE
# ------------------------------------------------
dark = st.session_state.dark_mode
colors = {
    "bg": "#0f172a" if dark else "#F9FAFB",
    "text": "#F9FAFB" if dark else "#111827",
    "bot_bg": "#1E293B" if dark else "#E5E7EB",
    "user_bg": "#2563EB",
    "input_bg": "#1E293B" if dark else "#FFFFFF",
    "border": "#3B82F6",
    "muted": "#9CA3AF" if dark else "#4B5563",
}

st.markdown(f"""
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: {colors['bg']};
            color: {colors['text']};
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
            background-color: {colors['user_bg']};
            color: white;
            text-align: right;
            border-bottom-right-radius: 4px;
        }}
        .bot-msg {{
            background-color: {colors['bot_bg']};
            color: {colors['text']};
            border-bottom-left-radius: 4px;
        }}
        .thinking {{
            font-style: italic;
            color: #9CA3AF;
            animation: blink 1.2s infinite;
        }}
        @keyframes blink {{ 50% {{ opacity: 0.4; }} }}
        .input-box {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 700px;
            background-color: {colors['input_bg']};
            border: 2px solid {colors['border']};
            border-radius: 25px;
            padding: 8px 20px;
        }}
        .stTextInput > div > div > input {{
            color: {colors['text']} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown(f"""
    <div style="text-align:center;">
        <h1>⚖️ Indian Legal AI Assistant</h1>
        <p style="color:{colors['muted']};">
            Ask about IPC, CrPC, or NIA — get precise, section-wise answers with citations.
        </p>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    show_citations = st.toggle("📜 Show Citations", value=st.session_state.show_citations)

    if dark_mode != st.session_state.dark_mode or show_citations != st.session_state.show_citations:
        st.session_state.dark_mode = dark_mode
        st.session_state.show_citations = show_citations
        st.rerun()

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------
# STREAM RESPONSE
# ------------------------------------------------
def generate_stream_response(query, context, show_citations=True):
    """
    Stream a high-quality legal answer (token-by-token) using Ollama.
    - query: user question
    - context: concatenated retrieved sections (must contain act & section info)
    - show_citations: whether to force explicit citations in the answer
    - model_name: Ollama model to use (e.g., "llama3.1")
    Returns the final generated text.
    """
    import time
    import streamlit as st
    import ollama

    # Safety / fallback
    if not query or not str(query).strip():
        return "⚠️ Empty query."

    # Short instruction for citation style
    citation_instruction = (
        "Include exact Act names and section numbers inline (e.g., 'IPC §52A')."
        if show_citations
        else "Do not include direct section numbers; give plain-English guidance."
    )

    # System message sets role & high-level constraints
    system_message = (
        "You are an expert Indian legal assistant (Supreme-Court-level) named 'Lex Legal AI'. "
        "Your primary goal is to provide highly accurate, context-based answers to Indian legal questions "
        "using only the provided LEGAL CONTEXT. "
        "However, if the user greets you or asks a general, personal, or unrelated question "
        "(like 'Hello', 'Who are you?', 'How are you?', 'Tell me a joke', etc.), "
        "you must respond naturally and conversationally as a friendly assistant — not in legal format. "
        "You may include light humor or empathy in such general replies, but remain professional. "
        "Never hallucinate or invent legal facts or sections. "
        "If the legal context is missing or incomplete, clearly state that and ask one short clarifying question. "
        "Always maintain a warm, respectful, and helpful tone."
    )

    user_prompt = f"""
    You are an Indian legal assistant with deep expertise in IPC, CrPC, and NIA.

    LEGAL CONTEXT:
    {context if context.strip() else '[NO CONTEXT PROVIDED]'}

    USER QUESTION:
    {query}

    ### RESPONSE RULES:
    1. If the user's query is **general, personal, or unrelated to law** (e.g., greetings, small talk, AI-related questions, etc.):
    → Respond casually and conversationally in a single paragraph.  
    → Example: “Hey there! I’m Lex Legal AI, your virtual Indian legal assistant. How can I help you today?”
    → Do **not** include any legal or structured response format.

    2. If the user's query is **legal** or **related to Indian law**:
    → Follow the structure below strictly.
    → Use **only** the provided LEGAL CONTEXT for legal facts.
    → If the context lacks sufficient data, say:
        "Insufficient context: [what’s missing]" and ask one short clarifying question.
    → {citation_instruction}
    → Keep responses factually correct, clear, and conversational — avoid complex legal jargon.

    ---

    ### OUTPUT FORMAT (for legal queries only):

    **Legal Act & Section:** [Act name and section(s) from context]  
    **Key Provisions:** [Brief bullet points of the main rules]  
    **Detailed Explanation:** [Plain, clear, accurate legal explanation referencing context]  
    **Practical Implications:** [How this applies in real scenarios]  
    **Related Provisions:** [Any connected sections or acts]  
    [One short clarifying question or leave it]  
    """

    response_text = ""
    stream = ollama.chat(model="llama3.1", messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ], stream=True)
    message_placeholder = st.empty()

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            response_text += content
            message_placeholder.markdown(f'<div class="chat-bubble bot-msg">{response_text}</div>', unsafe_allow_html=True)
            time.sleep(0.03)

    return response_text.strip()

# ------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    st.markdown(f'<div class="chat-bubble {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
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
# GENERATE ANSWER
# ------------------------------------------------
if st.session_state.is_generating:
    with st.spinner("💭 Thinking... analyzing your question..."):
        query = st.session_state.messages[-1]["content"]

        # Retrieve top matching legal references
        try:
            results = retrieve(query, top_k=3)
        except Exception as e:
            results = []
            # st.error(f"Retrieval Error: {str(e)}")

        context = "\n\n".join([
            f"{r['act']} Section {r['section']}: {r['title']}\n{r['text']}"
            for r in results
        ]) if results else "No relevant legal references found."

        answer = generate_stream_response(query, context, st.session_state.show_citations)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.is_generating = False
        st.rerun()
