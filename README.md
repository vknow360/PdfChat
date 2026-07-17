# PDF Chatbot 📄🤖

A full-stack conversational AI application built with Streamlit and LangChain that allows you to upload, process, and chat with your PDF documents seamlessly.

## ✨ Features

- **Document Ingestion**: Upload any PDF document directly through the UI.
- **Advanced RAG Architecture**: 
  - Text chunking via `RecursiveCharacterTextSplitter`
  - High-quality vector embeddings using HuggingFace (`BAAI/bge-small-en-v1.5`)
  - Lightning-fast vector search with locally stored FAISS indexes.
- **Conversational Memory**: Automatically manages chat history and summarizes older context (after 20 messages) to preserve token limits while maintaining context.
- **LLM Powered**: Leverages the blazing-fast `llama-3.3-70b-versatile` model via Groq for high-quality answers.
- **Clean Architecture**: Modular codebase separated into `ingestion`, `chains`, and UI (`app`).

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://python.langchain.com/)
- **LLM Provider**: [Groq](https://groq.com/) (Llama 3.3 70B)
- **Embeddings**: [HuggingFace](https://huggingface.co/)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd PdfChatbot
```

### 2. Install dependencies
Ensure you have Python 3.9+ installed, then install the required packages:
```bash
pip install streamlit langchain langchain-groq langchain-huggingface langchain-community faiss-cpu python-dotenv pypdf
```

### 3. Environment Variables
Create a `.env` file in the root directory based on the provided `.env.example`:
```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
```

### 4. Run the Application
Start the Streamlit development server:
```bash
streamlit run main.py
```
The app will open automatically in your browser at `http://localhost:8501`.

## 📁 Project Structure

```text
PdfChatbot/
├── main.py              # Application entry point
├── .env                 # Environment variables (API keys)
├── .env.example         # Template for environment variables
├── src/
│   ├── app.py           # Streamlit UI and app logic
│   ├── chains.py        # LangChain pipelines and prompts
│   ├── config.py        # Model and embedding configurations
│   ├── ingestion.py     # PDF processing and FAISS indexing
│   └── prompts.py       # Prompt templates (Chat & Summarization)
├── uploads/             # Temporarily stores uploaded PDFs
└── faiss_index/         # Stores the persistent vector databases
```

## 🧠 How It Works

1. **Upload**: You upload a PDF via the Streamlit interface.
2. **Process**: The PDF is saved to `/uploads` and parsed into text using `PyPDFLoader`.
3. **Chunk & Embed**: The text is split into chunks of 500 characters and embedded using HuggingFace.
4. **Index**: The embeddings are stored in a FAISS vector database inside `/faiss_index` for fast retrieval.
5. **Chat**: When you ask a question, the app retrieves the most relevant chunks from FAISS, injects them into a prompt alongside your chat history, and streams the answer back from Groq's Llama 3 model.

---
*Built with LangChain & Streamlit.*
