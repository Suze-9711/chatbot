import streamlit as st
import json
import requests
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from llama_cpp import Llama
from pathlib import Path
import huggingface_hub
import uuid

st.set_page_config(page_title="Chatbot", layout="centered")

# --- Model Info ---
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"

downloaded_path = huggingface_hub.hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    local_dir=".",
    local_dir_use_symlinks=False
)

# --- Load and Prepare Documents ---
docs = []
file_path = "RAG.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        docs.append(Document(page_content=entry["answer"], metadata={"title": entry.get("question", "")}))

cleaned_docs = [doc for doc in docs if len(doc.page_content.strip()) > 50 and "Question:" not in doc.page_content]
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
split_docs = splitter.split_documents(cleaned_docs)
texts_to_embed = [doc.page_content for doc in split_docs]

# --- Embed Documents ---
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(texts_to_embed, show_progress_bar=True)

# --- FAISS Indexing ---
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
document_lookup = {i: texts_to_embed[i] for i in range(len(texts_to_embed))}

# --- Retrieve and Ask ---
def retrieve_relevant_context(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [document_lookup[i] for i in indices[0]]

def ask_llm(query):
    context = "\n\n".join(retrieve_relevant_context(query))

    history_prompt = ""
    for role, text in st.session_state.chat_history:
        history_prompt += f"{role}: {text}\n"

    prompt = (
        f"Use the following context to answer the user's question:\n\n{context}\n\n"
        f"Conversation so far:\n{history_prompt}"
        f"User: {query}\n"
        f"Chat Bot:"
    )

    result = llm(prompt, max_tokens=512)
    return result["choices"][0]["text"].strip()

# --- Google Sheets Logging ---
def save_to_google_sheets(question, answer):
    url = "https://script.google.com/macros/s/AKfycby88z9hwW2aULsDuJ8rR4w9GCO9Bnb9x9oDZlViN99K15tyFjUoCybR2J0dcz_u-oAFWQ/exec"
    payload = {
        "session_id": st.session_state.session_id,
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Failed to log to Google Sheets:", e)

# --- Streamlit App ---
with st.spinner("Loading model..."):
    llm = Llama(
        model_path=str(downloaded_path),
        n_ctx=512,
        n_threads=4,
    )

st.title("Chatbot")

# --- Initialize chat history & session ID ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Clear Chat Button ---
if st.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# --- Input box ---
query = st.text_input("Ask a question...")

if query:
    with st.spinner("Thinking..."):
        answer = ask_llm(query)

    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Chat Bot", answer))
    save_to_google_sheets(query, answer)

# --- Display conversation ---
for role, text in st.session_state.chat_history:
    st.markdown(f"**{role}:** {text}")
