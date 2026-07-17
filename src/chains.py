from langchain_core.runnables import RunnableLambda

def format_docs(docs : list) -> str:
    return "\n".join(f"Page {doc.metadata['page']}\n{doc.page_content}\n" for doc in docs)

def format_messages(messages) -> str:
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


def build_chat_chain(retriever, model, prompt):
    """Builds and returns the main chat chain"""
    
    get_question = lambda x : x['question']
    return (
        {
            "context" : get_question | retriever | format_docs,
            "history": lambda x: format_messages(x['messages'][:-1]),
            "question" : get_question
        }
        | prompt
        | model
    )

def build_summary_chain(prompt, model, parser):
    """Builds and returns the summary chain"""
    
    return (prompt | model | parser)