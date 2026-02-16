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

def get_answer_generator(context, question):
    """
    Streams the answer from OpenAI chunk by chunk.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def contextualize_question(chat_history: list, latest_question: str):
    """
    Uses the chat history to rewrite the latest question 
    so it can be understood without context.
    """
    if not chat_history:
        return latest_question

    system_prompt = """You are a helpful assistant. 
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

    # Format history for the model
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append(msg)
    
    messages.append({"role": "user", "content": latest_question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    standalone_question = response.choices[0].message.content
    print(f"DEBUG REPHRASE: '{latest_question}' -> '{standalone_question}'")
    return standalone_question