import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when required production configuration is missing or invalid."""


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    pinecone_api_key: str | None
    pinecone_index_name: str = "medical-chatbot"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_dimension: int = 384
    model_path: Path = Path("model/llama-2-7b-chat.ggmlv3.q4_0.bin")
    model_type: str = "llama"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.3
    retrieval_k: int = 2
    max_question_chars: int = 2000

    @classmethod
    def from_env(cls, base_dir: Path) -> "AppConfig":
        model_path = Path(os.getenv("MODEL_PATH", "model/llama-2-7b-chat.ggmlv3.q4_0.bin"))
        if not model_path.is_absolute():
            model_path = base_dir / model_path

        return cls(
            base_dir=base_dir,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "medical-chatbot"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", os.getenv("PINECONE_API_ENV", "us-east-1")),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            model_path=model_path,
            model_type=os.getenv("MODEL_TYPE", "llama"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            debug=_bool_env("FLASK_DEBUG", False),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "2")),
            max_question_chars=int(os.getenv("MAX_QUESTION_CHARS", "2000")),
        )

    def validate_for_runtime(self) -> None:
        if not self.pinecone_api_key or self.pinecone_api_key == "replace-with-your-pinecone-api-key":
            raise ConfigError("PINECONE_API_KEY is required.")
        if not self.pinecone_index_name:
            raise ConfigError("PINECONE_INDEX_NAME is required.")
        if not self.model_path.exists():
            raise ConfigError(f"Model file was not found: {self.model_path}")
        if self.max_new_tokens < 1:
            raise ConfigError("MAX_NEW_TOKENS must be greater than zero.")
        if self.retrieval_k < 1:
            raise ConfigError("RETRIEVAL_K must be greater than zero.")
        if self.embedding_dimension < 1:
            raise ConfigError("EMBEDDING_DIMENSION must be greater than zero.")
