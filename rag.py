import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import math

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
DOCS_DIR = Path("data/course_docs")


class HashEmbeddings(Embeddings):
    """Small deterministic embedding model for a no-cost class demo.

    For a production system, replace this with OpenAIEmbeddings, HuggingFaceEmbeddings,
    or your university-approved embedding provider.
    """
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str):
        vec = [0.0] * self.dimensions
        for token in text.lower().replace("/", " ").replace(",", " ").split():
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings() -> Embeddings:
    return HashEmbeddings()


def build_vectorstore(persist_directory: str = CHROMA_DIR):
    loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_directory,
        collection_name="university_course_selection",
    )
    return vectorstore


def load_vectorstore(persist_directory: str = CHROMA_DIR):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
        collection_name="university_course_selection",
    )


def get_retriever(k: int = 5):
    return load_vectorstore().as_retriever(search_kwargs={"k": k})
