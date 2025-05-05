from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def get_ollama_llm():
    return Ollama(model="llama3")

def create_qa_chain(vectorstore):
    llm = get_ollama_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
