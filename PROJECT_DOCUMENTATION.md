# Medical Chatbot Project Documentation

## 1. Project Overview

This repository implements an end-to-end medical information chatbot using a local Llama 2 quantized model and Pinecone vector search.

The application combines:

- Flask for the web interface
- LangChain for retrieval-augmented answering
- Pinecone for vector storage and similarity search
- Hugging Face sentence-transformer embeddings
- CTransformers for loading the local Llama 2 GGML model
- PDF source documents from the `data/` folder

This system is designed for medical information retrieval. It must not be marketed as a diagnostic, emergency-care, or personalized treatment system.

## 2. Current Production Structure

- `app.py` - Flask app, endpoints, lazy QA-chain initialization
- `wsgi.py` - production server entry point using Waitress
- `store_index.py` - Pinecone index creation and document upload
- `src/config.py` - environment-driven configuration and validation
- `src/helper.py` - PDF loading, chunking, embeddings
- `src/prompt.py` - medical-information prompt
- `templates/chat.html` - self-contained chat UI
- `static/style.css` - chat UI styling
- `tests/` - automated tests
- `.env.example` - safe environment template
- `requirements.txt` - pinned dependency set
- `data/Medical_book.pdf` - source knowledge content
- `model/llama-2-7b-chat.ggmlv3.q4_0.bin` - local LLM model

## 3. How Indexing Works

1. `store_index.py` loads configuration from `.env`.
2. It creates the Pinecone serverless index if the index does not already exist.
3. It loads all PDFs from `data/`.
4. It splits loaded pages into chunks.
5. It embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`.
6. It uploads chunks to Pinecone.

Run indexing with:

```bash
python store_index.py
```

## 4. How Runtime Works

1. `app.py` creates the Flask app without loading the model at import time.
2. `/health` returns process health.
3. `/ready` validates required configuration and the local model file.
4. `/get` validates the request message.
5. On the first valid chat request, the app initializes embeddings, Pinecone vector search, prompt, local model, and `RetrievalQA`.
6. The answer is returned as JSON and rendered safely as text in the browser.

## 5. Production Fixes Applied

Security improvements:

- Removed hardcoded Pinecone API keys from source.
- Replaced the local `.env` secret with placeholders.
- Added `.env.example`.
- Removed old Bootstrap and jQuery CDN dependencies.
- Rendered user and model messages with `textContent` to prevent HTML injection.
- Added basic response security headers.
- Disabled Flask debug mode by default.

Runtime reliability improvements:

- Added config validation for API key, index name, model file, token count, retrieval count, and embedding dimension.
- Added lazy RAG initialization so tests and health checks do not load the 3.8 GB model.
- Added stable JSON errors for unavailable service states.
- Added PDF and chunk validation during indexing.
- Added automatic Pinecone index creation.
- Added focused pytest coverage.

Dependency improvements:

- Replaced conflicting old LangChain/Pinecone requirements with pinned compatible packages.
- Added explicit packages for `langchain-community`, `langchain-huggingface`, `langchain-pinecone`, and `langchain-text-splitters`.
- Added Waitress for production serving.
- Added pytest for verification.

Important: the previously exposed Pinecone key should be revoked in the Pinecone dashboard and replaced with a new key.

## 6. Configuration

Use `.env.example` as the template for `.env`.

Required:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `MODEL_PATH`

Common optional settings:

- `PINECONE_CLOUD`
- `PINECONE_REGION`
- `EMBEDDING_DIMENSION`
- `MODEL_TYPE`
- `EMBEDDING_MODEL`
- `HOST`
- `PORT`
- `FLASK_DEBUG`
- `MAX_NEW_TOKENS`
- `TEMPERATURE`
- `RETRIEVAL_K`
- `MAX_QUESTION_CHARS`

## 7. Runbook

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then set the real Pinecone key.

Build or rebuild the vector index:

```bash
python store_index.py
```

Run locally:

```bash
python app.py
```

Run for production:

```bash
python wsgi.py
```

Open:

```text
http://localhost:8080
```

Run tests:

```bash
pytest
```

## 8. Customer Delivery Checklist

- Replace `.env` placeholders with production secrets on the target server.
- Revoke the old exposed Pinecone key.
- Confirm `model/llama-2-7b-chat.ggmlv3.q4_0.bin` exists on the server.
- Run `python store_index.py` after adding final PDFs.
- Run `pytest`.
- Start with `python wsgi.py`.
- Check `GET /health` and `GET /ready`.
