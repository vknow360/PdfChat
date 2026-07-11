from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import streamlit as st

load_dotenv()

def create_chatbot(uploaded_file):
    loader = PyPDFLoader(file_path=uploaded_file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-small-en-v1.5")

    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local("faiss_index")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 30, "lambda_mult": 0.7}
    )
    return retriever


st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload pdf", type="pdf")
if uploaded_file and "retriever" not in st.session_state:
    with st.spinner("Processing PDF..."):
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.retriever = create_chatbot(save_path)

if "retriever" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    model = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7
            )

    prompt = PromptTemplate(
        template="""
        You are answering questions from a PDF.

        Use ONLY the provided context.

        If the user asks for a list,
        return every item found.

        Do not omit items.

        If the answer is not present,
        say "I don't know."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs : list) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context" : st.session_state.retriever | format_docs,
            "question" : RunnablePassthrough()
        }
        | prompt
        | model
    )

    
    query = st.chat_input("Ask question from pdf")
    if query:
        st.session_state.messages.append({"role": "user", "content" : query})
        st.chat_message("user").write(query)

        with st.spinner("Generating Response"):
            docs = st.session_state.retriever.invoke(query)
            
            answer = chain.invoke(
                query
            )

            st.session_state.messages.append({"role": "assistant", "content" : answer.content})
            st.chat_message("assistant").write(answer.content)

        

