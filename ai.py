import os
from dotenv import load_dotenv
from openai import OpenAI

# Load env once here
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    """Generates vector embeddings for a text string."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_answer(context: str, question: str):
    """Sends context + question to GPT-4o."""
    system_prompt = """You are a helpful assistant. 
    Answer the user's question strictly based on the provided context. 
    If the answer is not in the context, say 'I don't know'.
    The context might be messy text extracted from a PDF; do your best to clean it up."""

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return chat_response.choices[0].message.content