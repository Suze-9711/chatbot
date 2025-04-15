import os
import json
import hashlib
import requests
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

# --- Config ---
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycby88z9hwW2aULsDuJ8rR4w9GCO9Bnb9x9oDZlViN99K15tyFjUoCybR2J0dcz_u-oAFWQ/exec"


import gdown

def download_rag():
    if not os.path.exists("RAG.jsonl"):
        url = "https://drive.google.com/uc?id=1_MyPdp6xfwJcLtkroatCuOgm6GW410rj"
        gdown.download(url, "RAG.jsonl", quiet=False)

# --- Helper Functions ---
def encrypt(text):
    return hashlib.sha256(text.encode()).hexdigest()

def load_docs():
    docs = []
    with open("RAG.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            docs.append(Document(
                page_content=entry.get("answer", ""),
                metadata={"title": entry.get("question", "")}
            ))
    cleaned_docs = [doc for doc in docs if len(doc.page_content.strip()) > 50 and "Question:" not in doc.page_content]
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=20)
    return splitter.split_documents(cleaned_docs)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("faiss_index/index.faiss"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        download_rag()
        docs = load_docs()
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index")
        return db

def load_llm():
    return CTransformers(
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_file="tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        model_type="llama",  # ✅ Required for GGUF models
        config={
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
            "context_length": 2048,
            "stop": ["User:", "\nUser:", "\nBot:", "Assistant:"]
        }
    )

def format_chat_history(history):
    return "\n".join([f"User: {q}\nBot: {a}" for q, a in history])

def log_to_google_sheets(question, answer):
    encrypted_q = encrypt(question)
    encrypted_a = encrypt(answer)
    try:
        requests.post(WEBHOOK_URL, json={"question": encrypted_q, "answer": encrypted_a})
    except Exception as e:
        print(f"Error sending to Google Sheets: {e}")

# --- Streamlit Chat UI ---
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("✨ Kindstate Chatbot")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

QA_TEMPLATE = """
You are a kind and empathetic mental health assistant. Keep your responses under 100 words and focused on emotional support.

Context:
{context}

User: {question}
Assistant:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_TEMPLATE
)

@st.cache_resource

def setup_bot():
    db = load_vectorstore()
    llm = load_llm()
    chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
        document_variable_name="context"  # <--- Important line
    )
    return db, llm, chain


db, llm, chain = setup_bot()

# --- Chat History (top of screen) ---
st.markdown("### Chat History")
chat_container = st.container()
with chat_container:
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")

# --- Input Area (bottom of screen) ---
with st.container():
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", placeholder="I'm feeling overwhelmed...", label_visibility="collapsed")
        submit = st.form_submit_button("Send")

if submit and user_input:
    docs = db.similarity_search(user_input, k=4)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    answer = chain.run(input_documents=docs, question=user_input)
    if not answer:
        answer = "I'm here for you. Even when I’m quiet, you’re not alone. Could you share more with me?"

    st.session_state.history.append((user_input, answer))
    log_to_google_sheets(user_input, answer)
    st.rerun()
