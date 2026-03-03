import os
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader

# ---------------------- CONFIG ----------------------

DATA_PATH = "data"
PERSIST_DIRECTORY = "chroma_db"

# ---------------------- EMBEDDINGS ----------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------------- CREATE VECTORSTORE ----------------------

def create_vectorstore():

    all_documents = []

    # Read all PDFs from data folder
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file_name)
            reader = PdfReader(file_path)

            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            all_documents.append(Document(page_content=text, metadata={"source": file_name})    )

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(all_documents)

    # Create embeddings
    embeddings = get_embeddings()

    # Create Chroma DB
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    vectorstore.persist()

    return vectorstore

# ---------------------- LOAD VECTORSTORE ----------------------

def load_vectorstore():

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    return vectorstore