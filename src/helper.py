from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(data):
    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    loader = DirectoryLoader(str(data_path), glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF documents were loaded from: {data_path}")
    return documents


def text_split(extracted_data):
    if not extracted_data:
        raise ValueError("No documents were provided for text splitting.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    if not text_chunks:
        raise ValueError("No text chunks were created from the loaded documents.")
    return text_chunks


def download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
