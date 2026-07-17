from langchain_core.prompts import PromptTemplate

query_prompt = PromptTemplate(
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

summary_prompt = PromptTemplate(
    template="""
    You are a message summary expert.
    Summarize the following text in such a way that anyone can understand the essence in as minimum as possible words.

    Text to Summarize:
    {text}
    """,
    input_variables=["text"]
)