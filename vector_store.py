from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

import os

PERSIST_DIRECTORY = "db"

def create_vector_store(documents, model_name="nomic-embed-text"):
    print(f"🔢 Creating vector store with {len(documents)} chunks...")
    embedding = OllamaEmbeddings(model=model_name)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY
    )
    vectorstore.persist()
    print("✅ Vector store created and persisted.")
    return vectorstore

def load_vector_store(model_name="nomic-embed-text"):
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Vector store directory '{PERSIST_DIRECTORY}' does not exist.")
    
    embedding = OllamaEmbeddings(model=model_name)
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
