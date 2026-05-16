import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import AppConfig
from src.helper import download_hugging_face_embeddings, load_pdf, text_split

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def ensure_index(config: AppConfig) -> None:
    pc = Pinecone(api_key=config.pinecone_api_key)
    indexes = pc.list_indexes()
    existing_indexes = set(indexes.names()) if hasattr(indexes, "names") else {index["name"] for index in indexes}
    if config.pinecone_index_name in existing_indexes:
        return

    logger.info(
        "Creating Pinecone index '%s' in %s/%s",
        config.pinecone_index_name,
        config.pinecone_cloud,
        config.pinecone_region,
    )
    pc.create_index(
        name=config.pinecone_index_name,
        dimension=config.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region),
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    config = AppConfig.from_env(base_dir)
    if not config.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to build the Pinecone index.")

    ensure_index(config)
    extracted_data = load_pdf(base_dir / "data")
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings(config.embedding_model)

    logger.info("Uploading %s chunks to Pinecone index '%s'", len(text_chunks), config.pinecone_index_name)
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=config.pinecone_index_name,
        pinecone_api_key=config.pinecone_api_key,
    )
    logger.info("Index upload complete")


if __name__ == "__main__":
    main()
