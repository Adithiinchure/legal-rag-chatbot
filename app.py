import os

# Disable GPU BEFORE importing torch-based modules
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

import torch
torch.cuda.is_available = lambda: False

import streamlit as st
from rag_pipeline import create_vectorstore, load_vectorstore
from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME, PERSIST_DIRECTORY
import shutil
from pathlib import Path

st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

st.title("⚖️ Doc's Advice bot")
st.write("Ask legal questions from uploaded PDFs")

# ---------------------- PDF UPLOAD SECTION ----------------------

st.subheader("Upload Legal PDF")

uploaded_files = st.file_uploader(
    "Upload one or more legal PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)

    with st.spinner("Processing uploaded PDFs..."):
        for uploaded_file in uploaded_files:
            save_path = Path("data") / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Delete old vector DB silently
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except:
                pass  # No warning shown

        vectorstore = create_vectorstore()

# ---------------------- LOAD VECTOR DB ----------------------

client = Groq(api_key=GROQ_API_KEY)

try:
    if not os.path.exists(PERSIST_DIRECTORY):
        vectorstore = create_vectorstore()
    else:
        vectorstore = load_vectorstore()
except ValueError as e:
    st.error(f"⚠️ {str(e)}")
    st.info("📄 Please upload PDF files to get started.")
    st.stop()

# ---------------------- CHAT SECTION ----------------------

# ---------------------- CHAT SECTION ----------------------
# ---------------------- CHAT SECTION ----------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None

# ---------------------- SIDEBAR ----------------------

with st.sidebar:
    st.title("📚 Chat History")

    if st.session_state.chat_history:
        for i in range(0, len(st.session_state.chat_history), 2):
            question = st.session_state.chat_history[i]["content"]

            if st.button(question, key=f"chat_{i}"):
                st.session_state.selected_chat = i

    if st.button("🗑 Clear History"):
        st.session_state.chat_history = []
        st.session_state.selected_chat = None
        st.rerun()

# ---------------------- MAIN CHAT DISPLAY ----------------------

# If a previous chat is selected
if st.session_state.selected_chat is not None:
    index = st.session_state.selected_chat

    question = st.session_state.chat_history[index]["content"]
    answer = st.session_state.chat_history[index + 1]["content"]

    st.markdown("### ❓ Question")
    st.markdown(question)

    st.markdown("### 💬 Answer")
    st.markdown(answer)

else:
    # Show latest chat only
    if len(st.session_state.chat_history) >= 2:
        question = st.session_state.chat_history[-2]["content"]
        answer = st.session_state.chat_history[-1]["content"]

        st.markdown("### Question")
        st.markdown(question)

        st.markdown("### 💬  Answer")
        st.markdown(answer)

# ---------------------- CHAT INPUT ----------------------

if question := st.chat_input("Ask your legal question..."):

    docs = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a professional legal assistant.

Answer strictly from the given legal context.
If not found, say:
"Not found in provided documents."

Legal Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    # Save chat
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    st.session_state.selected_chat = None
    st.rerun()