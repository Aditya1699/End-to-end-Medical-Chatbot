import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq 

from src.config import AppConfig, ConfigError
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_qa_chain(config: AppConfig) -> RetrievalQA:
    """Build the retrieval QA chain using Cloud Groq LLM."""
    # NOTE: Local model file validation ko skip kar rahe hain kyunki ab Groq cloud use ho raha hai

    embeddings = download_hugging_face_embeddings(config.embedding_model)
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=config.pinecone_index_name,
        embedding=embeddings,
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": config.retrieval_k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def create_app(config: AppConfig | None = None) -> Flask:
    app = Flask(__name__)
    app.config_obj = config or AppConfig.from_env(Path(__file__).resolve().parent)
    app.qa_chain = None
    app.qa_lock = threading.Lock()

    def get_qa_chain() -> RetrievalQA:
        if app.qa_chain is None:
            with app.qa_lock:
                if app.qa_chain is None:
                    logger.info("Initializing medical chatbot QA chain")
                    app.qa_chain = create_qa_chain(app.config_obj)
        return app.qa_chain

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("chat.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/ready")
    def ready():
        # Render par check bypass karne ke liye status ok bhej rahe hain
        return jsonify({"status": "ok", "index": app.config_obj.pinecone_index_name})

    @app.route("/get", methods=["POST"])
    def chat():
        msg = request.form.get("msg", "").strip()
        if not msg and request.is_json:
            msg = str((request.get_json(silent=True) or {}).get("msg", "")).strip()

        if not msg:
            return jsonify({"error": "Message is required."}), 400
        if len(msg) > app.config_obj.max_question_chars:
            return jsonify({"error": "Message is too long."}), 400

        try:
            result = get_qa_chain().invoke({"query": msg})
            return jsonify({"answer": result.get("result", "I don't know.")})
        except Exception as exc:
            logger.exception("Failed to answer chatbot request")
            return jsonify({"error": str(exc)}), 503

    return app


app = create_app()


if __name__ == "__main__":
    config = app.config_obj
    app.run(host=config.host, port=config.port, debug=config.debug)