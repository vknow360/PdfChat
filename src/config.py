import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq

chat_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-small-en-v1.5")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))