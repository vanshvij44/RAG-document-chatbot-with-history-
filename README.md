# Conversational RAG Chatbot with PDF Uploads and Chat History

This project is a conversational RAG (Retrieval-Augmented Generation) chatbot that allows you to chat with your PDF documents. It maintains a chat history, enabling it to understand context from the ongoing conversation. The application is built using Streamlit, LangChain, and leverages the Groq API for fast language model inference and HuggingFace for text embeddings.

## Features

* **PDF Document Upload:** Upload one or more PDF files to be used as the knowledge base for the chatbot.
* **Conversational Interface:** A user-friendly chat interface powered by Streamlit.
* **Chat History:** The chatbot remembers the previous turns of the conversation to provide contextually relevant answers.
* **Retrieval-Augmented Generation (RAG):** Uses the RAG technique to retrieve relevant information from the uploaded documents to answer user questions.
* **Fast LLM Inference:** Integrates with the Groq API for high-speed responses from the Gemma2-9b-It language model.
* **State-of-the-art Embeddings:** Utilizes HuggingFace's `all-MiniLM-L6-v2` model for generating text embeddings.
* **Session Management:** Supports different chat sessions with unique session IDs.

## How it Works

1.  **PDF Processing:** When you upload PDF files, the application extracts the text from them.
2.  **Text Splitting and Embedding:** The extracted text is split into smaller chunks, and each chunk is converted into a numerical representation (embedding) using a HuggingFace model.
3.  **Vector Storage:** These embeddings are stored in a Chroma vector store for efficient retrieval.
4.  **History-Aware Retriever:** A history-aware retriever is created, which takes into account the chat history to reformulate the user's question into a standalone query.
5.  **Question Answering:** When you ask a question, the retriever finds the most relevant text chunks from the vector store. These chunks, along with the chat history and your question, are passed to the language model.
6.  **Response Generation:** The language model generates a concise answer based on the provided context. If the model doesn't know the answer, it will say so.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vanshvij44/rag-document-chatbot-with-history-.git](https://github.com/vanshvij44/rag-document-chatbot-with-history-.git)
    cd rag-document-chatbot-with-history-
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    You will need to create a `requirements.txt` file with the following content:
    ```
    streamlit
    langchain
    langchain-groq
    langchain-community
    langchain-huggingface
    langchain-core
    chromadb
    pypdf
    python-dotenv
    huggingface_hub
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your API keys:
    ```
    HF_API_KEY="your_hugging_face_api_key"
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open the application in your browser.**

3.  **Enter your Groq API key** in the text input field.

4.  **Upload your PDF files.**

5.  **Start chatting** with your documents!

## Dependencies

* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [Groq](https://groq.com/)
* [Hugging Face](https://huggingface.co/)
* [Chroma](https://www.trychroma.com/)
* [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdf)

