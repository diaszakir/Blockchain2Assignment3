import os
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIRECTORY = "vector_db"
DEFAULT_EMBEDDING_MODEL = "llama3"

def get_embedding_model(model_name=DEFAULT_EMBEDDING_MODEL):
    """Get embedding model based on availability"""
    try:
        # Try Ollama first
        return OllamaEmbeddings(model=model_name)
    except Exception as e:
        logger.warning(f"Failed to load Ollama embeddings: {e}")
        
        # Fallback to HuggingFace embeddings
        try:
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e2:
            logger.error(f"Failed to load HuggingFace embeddings: {e2}")
            raise Exception("No embedding model available. Please install Ollama or HuggingFace.")

def create_vector_store(documents, persist_directory=PERSIST_DIRECTORY):
    """Create a new vector store from documents"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Get embedding model
        embedding = get_embedding_model()
        
        logger.info("Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        
        # Persist to disk
        vectorstore.persist()
        logger.info(f"Vector store created successfully with {len(documents)} documents")
        
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise e

def load_vector_store(persist_directory=PERSIST_DIRECTORY):
    """Load an existing vector store from disk"""
    try:
        if not os.path.exists(persist_directory):
            logger.warning(f"Vector store directory {persist_directory} does not exist")
            return None
            
        # Get embedding model
        embedding = get_embedding_model()
        
        logger.info(f"Loading vector store from {persist_directory}...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        
        logger.info("Vector store loaded successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise e
        
def clear_vector_store(persist_directory=PERSIST_DIRECTORY):
    """Clear the vector store"""
    try:
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
            logger.info(f"Vector store at {persist_directory} cleared")
            return True
        return False
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        return False