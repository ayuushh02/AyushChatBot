"""
MAIN.PY - RAG Chatbot Application

What it does:
- Creates a web-based chatbot that can answer questions using your indexed documents
- Implements RAG (Retrieval-Augmented Generation) by finding relevant context before responding
- Maintains conversation history and streams responses in real-time

How it works:
1. When user asks a question:
   - Converts the question into an embedding using Gemini
   - Searches Pinecone for the 3 most similar documents
   - Combines the retrieved text as context
2. Builds a conversation with:
   - System message including the retrieved context
   - Previous conversation history
   - Current user question
3. Sends everything to OpenAI for a streaming response
4. Returns the AI's answer based on both the context and its general knowledge

Since our source files are small, each document is indexed as a complete unit without chunking.
The result is a chatbot that can answer questions about your specific documents
while still being able to have general conversations.
"""

import os
import gradio as gr
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from embedding import get_embedding

load_dotenv("../.env")

# Initialize clients
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def search_knowledge(query, top_k=3):
    """Search Pinecone for relevant chunks"""
    query_embedding = get_embedding(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    context = []
    for match in results['matches']:
        context.append(match['metadata']['text'])
    
    return "\n\n".join(context)

def chat(message, history):
    # Get relevant context from Pinecone
    context = search_knowledge(message)
    
    # Build messages with context
    system_message = f"""You are a helpful AI assistant with access to recent tech news. Use the following context to answer questions:

{context}

Answer based on the provided context when relevant, but you can also use your general knowledge for other questions."""

    messages = [{"role": "system", "content": system_message}]
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": message})
    
    stream = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=messages,
        stream=True
    )
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response

demo = gr.ChatInterface(chat, title="RAG Chatbot - Tech News Assistant")
demo.launch()