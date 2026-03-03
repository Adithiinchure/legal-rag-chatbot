import os

# Disable GPU to avoid torch meta tensor issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"




from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import DATA_PATH, PERSIST_DIRECTORY
from langchain_community.document_loaders import PyPDFLoader


# Force torch to CPU before any model loading
import torch
torch.cuda.is_available = lambda: False

# Workaround for meta tensor issue
def get_embeddings_safe():
    """Load embeddings with workaround for torch meta tensor issue"""
    try:
        # Use sentence_transformers with explicit settings to avoid meta tensors
        from sentence_transformers import SentenceTransformer
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu",
            trust_remote_code=True
        )
        
        class SafeEmbeddings:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts):
                try:
                    return self.model.encode(texts, convert_to_numpy=True).tolist()
                except (RuntimeError, NotImplementedError) as e:
                    # Fallback: encode one by one
                    embeddings = []
                    for text in texts:
                        try:
                            emb = self.model.encode([text], convert_to_numpy=True)[0].tolist()
                            embeddings.append(emb)
                        except Exception as inner_e:
                            # Skip problematic texts
                            print(f"Warning: Could not embed text: {str(inner_e)}")
                            # Return zeros if embedding fails
                            embeddings.append([0.0] * 384)  # MiniLM-L6-v2 has 384 dimensions
                    return embeddings
            
            def embed_query(self, text):
                try:
                    return self.model.encode(text, convert_to_numpy=True).tolist()
                except (RuntimeError, NotImplementedError):
                    # Return zeros if embedding fails
                    return [0.0] * 384
        
        return SafeEmbeddings(model)
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings: {str(e)}")

def create_vectorstore():

    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()
            documents.extend(docs)

    if not documents:
        raise ValueError(f"No PDF files found in {DATA_PATH}. Please add PDF files to the data directory.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    embeddings = get_embeddings_safe()

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    vectorstore.persist()
    return vectorstore


def load_vectorstore():

    embeddings = get_embeddings_safe()

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )