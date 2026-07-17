import os
from src.config import BASE_DIR
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader

FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

def create_retriever(uploaded_file:str, embeddings: Embeddings):
    file_name = os.path.splitext(os.path.basename(uploaded_file))[0]
    faiss_path = os.path.join(FAISS_DIR, file_name)

    vector_store = None
    if not os.path.exists(faiss_path):
        loader = PyPDFLoader(file_path=uploaded_file)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = splitter.split_documents(docs)
        
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(faiss_path)
    else:
        vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "lambda_mult": 0.7}
    )
    return retriever
