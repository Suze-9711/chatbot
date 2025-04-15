import streamlit as st
import json
import requests
import csv
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from llama_cpp import Llama
import os
import requests
from pathlib import Path

def download_from_gdrive(file_id, dest_path):
    if Path(dest_path).exists():
        print("Model already downloaded.")
        return

    print("Downloading model from Google Drive...")
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print("Download complete!")


model_path = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
download_from_gdrive("1YrYfjlVvYTQlPl7IGRY95_9CYZGYYsC1", model_path)

if not Path(model_path).exists():
    raise FileNotFoundError(f"Model not found after download: {model_path}")
else:
    print("Model file exists, loading...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=8,
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
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    result = llm(prompt, max_tokens=512)
    return result["choices"][0]["text"].strip()

# --- Save to Google Sheets (Apps Script Webhook) ---
def save_to_google_sheets(question, answer):
    url = "https://script.google.com/macros/s/AKfycby88z9hwW2aULsDuJ8rR4w9GCO9Bnb9x9oDZlViN99K15tyFjUoCybR2J0dcz_u-oAFWQ/exec"
    payload = {"question": question, "answer": answer, "timestamp": datetime.utcnow().isoformat()}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Failed to log to Google Sheets:", e)

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot")

query = st.text_input("Ask a question...")
if query:
    answer = ask_llm(query)
    st.markdown(f"** Chat Bot:** {answer}")
    save_to_google_sheets(query, answer)
