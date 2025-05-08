# 🇰🇿 AI Assistant for the Constitution of the Republic of Kazakhstan

## Overview

This project is a Minimum Viable Product (MVP) of an AI Assistant designed to answer questions related to the Constitution of the Republic of Kazakhstan. It leverages Large Language Models (LLMs) and vector databases to provide accurate and context-aware responses.

## Features

* **Chat Interface**: Interactive chat powered by Streamlit.
* **LLM Integration**: Utilizes model - Ollama - for natural language understanding.
* **Document Upload**: Users can upload one or multiple documents (PDF, DOCX, TXT).
* **Contextual Q\&A**: Answers questions based on the uploaded documents and the Constitution.
* **Vector Store**: Stores queries and answers using MongoDB for efficient retrieval.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/diaszakir/Blockchain2Assignment3.git
   cd Blockchain2Assignment3
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**:

   ```bash
   streamlit run main.py
   ```

2. **Interact with the Chatbot**:

   * Upload documents related to the Constitution.
   * Ask questions in the chat interface.
   * Receive context-aware answers.

## Project Structure

* `main.py`: Entry point for the Streamlit application.
* `app.py`: Contains the main application logic.
* `llm_service.py`: Handles interactions with the chosen LLM.
* `vector_store.py`: Manages storage and retrieval of embeddings.
* `document_loader.py`: Processes and loads uploaded documents.
* `chat_history.py`: Maintains the history of user interactions.
* `ollama_llm.py`: Specific implementation for the Ollama LLM.
* `requirements.txt`: Lists all Python dependencies.

## Screenshots



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Collaborators

Dias Zakir SE-2320

Anvar Tamabayev SE-2320

Danel Kanbakova SE-2320

## References

1. [RAG with ChromaDB and Ollama](https://medium.com/@arunpatidar26/rag-chromadb-ollama-python-guidefor-beginners-30857499d0a0)
2. [Streamlit File Uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
3. [LangChain Document Transformers](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)

