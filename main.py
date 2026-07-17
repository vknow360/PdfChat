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

def create_chatbot(uploaded_file:str):
    file_name = os.path.splitext(os.path.basename(uploaded_file))[0]

    embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-small-en-v1.5")
    vector_store = None
    if not os.path.exists(os.path.join("faiss_index", file_name)):
        loader = PyPDFLoader(file_path=uploaded_file)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = splitter.split_documents(docs)
        
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(os.path.join("faiss_index", file_name))
    else:
        vector_store = FAISS.load_local(os.path.join("faiss_index", file_name), embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "lambda_mult": 0.7}
    )
    return retriever

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

        Conversation:
        {history}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "history", "question"]
    )

def format_docs(docs : list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def format_messages(messages) -> str:
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-1]])

st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload pdf", type="pdf")
if uploaded_file:
    if "last_file" not in st.session_state:
        st.session_state.last_file = None
    
    if uploaded_file.name != st.session_state.last_file:
        with st.spinner("Processing PDF..."):
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state.retriever = create_chatbot(save_path)
            st.session_state.last_file = uploaded_file.name

if "retriever" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    retriever = st.session_state.retriever

    get_question = lambda x : x['question']

    chain = (
        {
            "context" : get_question | retriever | format_docs,
            "history": lambda x: format_messages(x['messages']),
            "question" : get_question
        }
        | prompt
        | model
    )

    
    query = st.chat_input("Ask question from pdf")
    if query:
        st.session_state.messages.append({"role": "user", "content" : query})
        st.chat_message("user").write(query)

        with st.spinner("Generating Response"):

            answer = chain.invoke(
                {
                    'question': query,
                    'messages': st.session_state.messages
                }
            )

            st.session_state.messages.append({"role": "assistant", "content" : answer.content})
            st.chat_message("assistant").write(answer.content)