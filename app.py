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

# Load environment configurations
load_dotenv()

# Global logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_qa_chain(config: AppConfig) -> RetrievalQA:
    """
    Builds the retrieval QA chain using Cloud Groq LLM eagerly at app startup.
    This prevents Render's HTTP 30-second timeout on the first user message.
    """
    logger.info("=============================================================")
    logger.info("🚀 TRIGGERING EAGER INITIALIZATION OF MEDICAL QA CHAIN...")
    logger.info("=============================================================")
    
    # Explicitly fetching core keys to avoid silent environment drops
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

    if not pinecone_api_key or not groq_api_key:
        logger.error("❌ CRITICAL: Missing API Keys in environment variables!")
        raise ConfigError("Pinecone or Groq API keys are not properly set in the cloud environment.")

    # 1. Download/Load Embeddings model upfront on server spin-up
    logger.info("🔄 Step 1/3: Bootstrapping Hugging Face embeddings onto RAM...")
    embeddings = download_hugging_face_embeddings(config.embedding_model)
    
    # 2. Bind Vector Store connection hooks
    logger.info("🔄 Step 2/3: Connecting to Pinecone Index: '%s'...", config.pinecone_index_name)
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=config.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )

    # 3. Construct prompt templates and cloud inference routing
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    logger.info("🔄 Step 3/3: Initializing Cloud Groq Core Engine (Model: %s)...", groq_model)
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=groq_model,
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
    )

    logger.info("✅ SUCCESS: Medical RAG components fully loaded into memory!")
    logger.info("=============================================================")
    
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
    
    # Thread-safe pipeline references
    app.qa_chain = None
    app.qa_lock = threading.Lock()

    # EAGER LOADING STRATEGY: 
    # Force initialisation during application assembly rather than on the first request context.
    try:
        app.qa_chain = create_qa_chain(app.config_obj)
    except Exception as init_exc:
        logger.error("❌ CRITICAL: App assembly failed to pre-cache RAG pipeline: %s", init_exc)
        app.qa_chain = None

    def get_qa_chain() -> RetrievalQA:
        """Fallback dynamic gateway if background orchestration requires re-instantiation."""
        if app.qa_chain is None:
            with app.qa_lock:
                if app.qa_chain is None:
                    logger.warning("🔄 Pipeline was uninitialized. Attempting hot reload context...")
                    app.qa_chain = create_qa_chain(app.config_obj)
        return app.qa_chain

    @app.after_request
    def add_security_headers(response):
        """Enforces safe header architectures required for SaaS application compliance."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.get("/")
    def index():
        return render_template("chat.html")

    @app.get("/health")
    def health():
        """Used by cloud balancers to verify microservice process viability."""
        return jsonify({"status": "ok"}), 200

    @app.get("/ready")
    def ready():
        """Bypasses cold-start health check blocks on hosting setups."""
        if app.qa_chain is None:
            return jsonify({"status": "initializing", "detail": "Models are loading into RAM."}), 503
        return jsonify({"status": "ok", "index": app.config_obj.pinecone_index_name}), 200

    @app.route("/get", methods=["POST"])
    def chat():
        """Handles user inquiries directly from RAM without database look-up or loading lag."""
        msg = request.form.get("msg", "").strip()
        if not msg and request.is_json:
            msg = str((request.get_json(silent=True) or {}).get("msg", "")).strip()

        if not msg:
            return jsonify({"error": "Message content cannot be blank."}), 400
        if len(msg) > app.config_obj.max_question_chars:
            return jsonify({"error": f"Message context limit exceeded. Maximum allowed characters: {app.config_obj.max_question_chars}"}), 400

        # Fast failure check to prevent spinning up timeout requests
        if app.qa_chain is None:
            return jsonify({"error": "System is currently optimizing memory models. Please refresh and try again in 15 seconds."}), 503

        try:
            # Direct calculation hit on our pre-cached engine
            result = get_qa_chain().invoke({"query": msg})
            return jsonify({"answer": result.get("result", "Information could not be parsed from medical database.")})
        except Exception as exc:
            logger.exception("❌ Error encountered during query computation trace")
            return jsonify({"error": f"An infrastructure error occurred: {str(exc)}"}), 503

    return app


app = create_app()


if __name__ == "__main__":
    config = app.config_obj
    app.run(host=config.host, port=config.port, debug=config.debug)