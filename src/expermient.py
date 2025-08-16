import os
import io
import time
import threading
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)


load_dotenv()
st.set_page_config(page_title="Amlgo Labs – RAG Chatbot", page_icon="🤖", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Pick a lightweight model if you don't have a GPU.")
    model_name = st.selectbox(
        "Instruction model",
        options=[
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2-1.5B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 2048, 512, 32)
    top_k = st.slider("Retriever top_k", 1, 10, 4, 1)
    chunk_size = st.slider("Chunk size", 300, 2000, 900, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)

    persist_base = st.text_input("Chroma persist dir", value=".chroma_store")
    session_id = st.text_input("Session ID", value="default_session")

    st.divider()
    st.caption("Optional: PEFT/LoRA adapter path (if you've fine‑tuned)")
    lora_dir = st.text_input("LORA_ADAPTER_DIR", value="")


@st.cache_resource(show_spinner=True)
def load_local_model(model_name: str, lora_dir: str = ""):
    """Load tokenizer & model; supports CPU/GPU and optional LoRA."""
    device_map = "auto"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure chat template exists; many instruct models provide it. If not, we build our own prompt later.

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=torch_dtype,
        )


    if lora_dir and os.path.isdir(lora_dir):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
        except Exception as e:
            st.warning(f"Could not load LoRA adapter from {lora_dir}: {e}")

    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=True)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def make_vectorstore(docs, persist_dir: str):
    embeddings = get_embeddings()
    vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    # vs.persist()
    return vs



SYSTEM_PROMPT = (
    "You are a helpful legal/policy assistant. Answer using ONLY the provided context. "
    "If the answer isn't in the context, say you don't know. Be concise and cite sources as [S1], [S2] etc."
)


def build_rag_prompt(context_blocks: List[str], user_msg: str, history: List[Dict[str, str]]) -> str:
    # Convert history (list of {role, content}) to simple transcript lines
    hist_lines = []
    for turn in history[-6:]:  # last 6 messages to keep prompt small
        role = "User" if turn["role"] == "user" else "Assistant"
        hist_lines.append(f"{role}: {turn['content']}")
    history_txt = "\n".join(hist_lines)

    context_assembled = "\n\n".join([f"[S{i+1}]\n{c}" for i, c in enumerate(context_blocks)])

    prompt = f"""
{SYSTEM_PROMPT}

Conversation so far:
{history_txt}

Context documents:
{context_assembled}

User: {user_msg}
Assistant:
""".strip()
    return prompt



@torch.inference_mode()
def stream_generate(tokenizer, model, prompt: str, temperature: float, top_p: float, max_new_tokens: int):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(0.01, float(temperature)),
        top_p=float(top_p),
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for token in streamer:
        yield token
    thread.join()



st.title("Fine‑Tuned/Instruction RAG Chatbot")
st.caption("Upload policy/legal PDFs → ask questions → get streaming answers with citations.")

# Session state
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "sources" not in st.session_state:
    st.session_state.sources = []

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📄 Upload documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )
    process_btn = st.button("🔧 Process & Index")

with col_right:
    st.subheader("📚 Current index")
    if st.session_state.vectorstore:
        stats = st.session_state.vectorstore._collection.count()
        st.metric("Chunks in index", stats)
        st.write("Persist dir:", os.path.join(persist_base, session_id))
    else:
        st.info("No index yet. Upload and click 'Process & Index'.")

# Indexing flow
if process_btn and uploaded_files:
    with st.status("Processing PDFs → splitting → embedding → indexing...", expanded=False) as status:
        docs = []
        for uf in uploaded_files:
            # save to temp buffer and load via PyPDFLoader
            path = os.path.join("/tmp", f"{time.time_ns()}_{uf.name}")
            with open(path, "wb") as f:
                f.write(uf.getvalue())
            loader = PyPDFLoader(path)
            raw_docs = loader.load()
            # split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "]
            )
            splits = splitter.split_documents(raw_docs)
            docs.extend(splits)
        status.update(label="Building vector store...", state="running")
        persist_dir = os.path.join(persist_base, session_id)
        os.makedirs(persist_dir, exist_ok=True)
        vs = make_vectorstore(docs, persist_dir=persist_dir)
        st.session_state.vectorstore = vs
        status.update(label="Index ready!", state="complete")

# Load model lazily
with st.spinner("Loading model (first time takes a bit)..."):
    tokenizer, model = load_local_model(model_name=model_name, lora_dir=lora_dir)

st.divider()
st.subheader("💬 Chat")

# Render past chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_msg = st.chat_input("Ask a question about your documents…")

if user_msg:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and index documents first.")
    else:
        # Show user message immediately
        st.session_state.history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Retrieve context
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retriever.get_relevant_documents(user_msg)
        context_blocks = []
        sources_for_display = []
        for i, d in enumerate(retrieved_docs):
            # keep short contexts
            snippet = d.page_content.strip()
            # Tag a short source label
            meta = d.metadata or {}
            source_label = meta.get("source", meta.get("file_path", "document"))
            page = meta.get("page", None)
            if page is not None:
                source_label = f"{os.path.basename(source_label)}#p{page+1}"
            sources_for_display.append(source_label)
            context_blocks.append(snippet)

        st.session_state.sources = sources_for_display

        # Build prompt with history & context
        prompt = build_rag_prompt(context_blocks, user_msg, st.session_state.history)

        # Stream the model's response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            for token in stream_generate(
                tokenizer,
                model,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            ):
                streamed_text += token
                placeholder.markdown(streamed_text)

            # After streaming completes, append citations
            if st.session_state.sources:
                cited = "\n\n" + "**Sources:** " + ", ".join(
                    [f"[S{i+1}] {s}" for i, s in enumerate(st.session_state.sources)]
                )
                streamed_text += cited
                placeholder.markdown(streamed_text)

        # Save assistant turn
        st.session_state.history.append({"role": "assistant", "content": streamed_text})

st.divider()
with st.expander("🔍 Debug – last retrieved chunks"):
    if st.session_state.sources:
        for i, src in enumerate(st.session_state.sources, start=1):
            st.write(f"[S{i}] {src}")
    else:
        st.caption("No retrievals yet.")

with st.expander("🧠 System prompt"):
    st.code(SYSTEM_PROMPT)
