# Medical Chatbot

Production-ready Flask RAG chatbot for answering questions from medical PDF content using Flask, LangChain, Pinecone, Hugging Face embeddings, and a local CTransformers Llama 2 GGML model.

This application is for medical information retrieval only. It must not be presented as a diagnostic, emergency-care, or personalized treatment system.

## Requirements

- Python 3.10 or 3.11
- A Pinecone API key
- The local model file at `model/llama-2-7b-chat.ggmlv3.q4_0.bin`
- Source PDFs in `data/`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set your real Pinecone key:

```ini
PINECONE_API_KEY=your-real-key
PINECONE_INDEX_NAME=medical-chatbot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
MODEL_PATH=model/llama-2-7b-chat.ggmlv3.q4_0.bin
FLASK_DEBUG=false
```

## Build the Vector Index

Run this once after adding or changing PDFs:

```bash
python store_index.py
```

`store_index.py` creates the Pinecone serverless index if it does not exist, then uploads the document chunks.

## Run Locally

```bash
python app.py
```

Open `http://localhost:8080`.

## Production Run

Use Waitress instead of Flask debug server:

```bash
python wsgi.py
```

Health endpoints:

- `GET /health` checks the web process.
- `GET /ready` validates required runtime configuration and model file presence.

## Test

```bash
pytest
```

## Security Notes

- No API keys are hardcoded in source.
- `.env` is ignored by Git.
- The old exposed Pinecone key should be revoked in Pinecone and replaced with a new key.
- The chat UI escapes user and model text before rendering.
- Flask debug mode is disabled by default.
