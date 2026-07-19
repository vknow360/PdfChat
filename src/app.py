import uuid
import hashlib
import threading
from src.chains import build_chat_chain, format_messages
from src.chains import build_summary_chain
from src.config import chat_model, embeddings, BASE_DIR
from src.prompts import query_prompt, summary_prompt
from langchain_core.output_parsers import StrOutputParser
import os
from fastapi import FastAPI, UploadFile

from src.ingestion import create_retriever

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

lock_a = threading.Lock()

conversations = {}

class Conversation():
    def __init__(self, file_hash):
        self.id = uuid.uuid4().__str__()
        self.messages = []
        self.file_hash = file_hash
        self.retriever = create_retriever(os.path.join(UPLOAD_DIR, f"{file_hash}.pdf"), embeddings)

    def __str__(self):
        return self.id

parser = StrOutputParser()


app = FastAPI()

@app.post("/upload")
def upload_file(file: UploadFile):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    data = file.file.read()
    file_hash = hashlib.md5(data).hexdigest()
    file_path = os.path.join(UPLOAD_DIR, f"{file_hash}.pdf")
    
    with open(file_path, "wb") as f:
        f.write(data)

    conversation = Conversation(file_hash)
    conversations[conversation.id] = conversation
    return conversation.id

@app.post("/chat/{id}")
def chat(id: str, msg: str):
    conversation = conversations.get(id)
    if not conversation:
        return {"error": "Conversation not found"}

    with lock_a:
        if len(conversation.messages) > 20:
            summary_chain = build_summary_chain(summary_prompt, chat_model, parser)
            summary_content = summary_chain.invoke({
                "text": format_messages(conversation.messages[:-10])
            })
            conversation.messages = [
                {"role": "system", "content": summary_content} , *conversation.messages[-10:]
            ]

    conversation.messages.append({"role": "user", "content": msg})

    query_chain = build_chat_chain(conversation.retriever, chat_model, query_prompt)
    answer = query_chain.invoke(
        {
            'question': msg,
            'messages': conversation.messages
        }
    )
    conversation.messages.append({"role": "assistant", "content" : answer})
    return answer
