from src.chains import build_chat_chain, format_messages
from src.chains import build_summary_chain
from src.config import chat_model, embeddings, BASE_DIR
from src.prompts import query_prompt, summary_prompt
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st

from src.ingestion import create_retriever

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

@st.cache_resource
def get_model():
    return chat_model

@st.cache_resource
def get_embeddings():
    return embeddings

model = get_model()
embeddings = get_embeddings()

parser = StrOutputParser()


def run_app():
    st.title("PDF Chatbot")

    uploaded_file = st.file_uploader("Upload pdf", type="pdf")
    if uploaded_file:
        if "last_file" not in st.session_state:
            st.session_state.last_file = None
        
        if uploaded_file.name != st.session_state.last_file:
            try:
                with st.spinner("Processing PDF..."):
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.session_state.retriever = create_retriever(save_path, embeddings)
                st.session_state.last_file = uploaded_file.name
            except Exception as e:
                st.error(f"Error processing file: {e}")

    if "retriever" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        retriever = st.session_state.retriever
        
        query = st.chat_input("Ask question from pdf")
        if query:
            st.session_state.messages.append({"role": "user", "content" : query})
            st.chat_message("user").write(query)

            with st.spinner("Generating Response"):

                if len(st.session_state.messages) > 20:
                    
                    summary_chain = build_summary_chain(summary_prompt, model, parser)
                    summary_content = summary_chain.invoke({
                        "text": format_messages(st.session_state.messages[:-10])
                    })
                    st.session_state.messages = [
                        {"role": "assistant", "content": summary_content} , *st.session_state.messages[-10:]
                    ]

                query_chain = build_chat_chain(retriever, model, query_prompt)
                answer_stream = query_chain.stream(
                    {
                        'question': query,
                        'messages': st.session_state.messages
                    }
                )

                response = st.chat_message("assistant").write_stream(answer_stream)
                st.session_state.messages.append({"role": "assistant", "content" : response})
