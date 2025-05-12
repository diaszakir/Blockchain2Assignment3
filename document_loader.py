import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CONSTITUTION_FILE = "documents/constitution_kazakhstan.pdf"

def load_constitution():
    
    if not os.path.exists(CONSTITUTION_FILE):
        raise FileNotFoundError(f"File not found")

    try:
        if CONSTITUTION_FILE.endswith(".pdf"):
            loader = PyPDFLoader(CONSTITUTION_FILE)
        elif CONSTITUTION_FILE.endswith(".docx"):
            loader = Docx2txtLoader(CONSTITUTION_FILE)
        elif CONSTITUTION_FILE.endswith(".txt"):
            loader = TextLoader(CONSTITUTION_FILE, encoding="utf-8")
        else:
            raise ValueError("Error")
        
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(CONSTITUTION_FILE)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return text_splitter.split_documents(docs)
    
    except Exception as e:
        print(f"Loading error: {e}")
        raise e


def load_and_split_documents(file_paths):
    documents = []

    for path in file_paths:
        try:
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.lower().endswith(".docx"):
                loader = Docx2txtLoader(path)
            elif path.lower().endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                print(f"Incorrect format: {path}")
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = os.path.basename(path)

            documents.extend(docs)

        except Exception as e:
            print(f"Loading error {path}: {e}")
            continue

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    return []
