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

    system_prompt = """You are a helpful assistant for a Resume/CV Analysis system. 
    Given a chat history and the latest user question, formulate a standalone question.
    Context: The user is asking about a candidate's professional background.
    Do NOT add external assumptions (like movies, stories, or fiction). 
    If the name 'Jafar' is used, assume it refers to the candidate in the documents.
    Return only the reformulated question."""

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

def classify_intent(chat_history: list, question: str) -> str:
    """
    Decides if the question requires looking up the database or is just general chat.
    Returns: 'search' or 'chat'
    """
    system_prompt = """You are a query router. Your job is to classify the user's intent.
    
    - If the user asks about specific people (e.g. 'Jafar'), internal documents, skills, resumes, or specific facts not in general knowledge -> Return 'search'.
    - If the user says 'hello', 'thanks', 'help', or asks general questions (e.g. 'What is Python?') -> Return 'chat'.
    
    Output ONLY one word: 'search' or 'chat'. Do not output anything else."""

    # Use the last few messages for context
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add brief history (last 2 turns) to understand context (e.g. "What about him?")
    for msg in chat_history[-2:]: 
        messages.append(msg)
    
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o-mini", # Use the fastest/cheapest model
        messages=messages,
        temperature=0 # Be deterministic
    )
    
    decision = response.choices[0].message.content.strip().lower()
    print(f"🧠 ROUTER DECISION: {decision.upper()}")
    return decision

def get_general_chat_generator(chat_history: list, question: str):
    """
    Handles general conversation without looking at the vector DB.
    """
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    
    # Pass full history so it can have a conversation
    for msg in chat_history:
        messages.append(msg)
        
    messages.append({"role": "user", "content": question})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content