import os
from google import genai
from dotenv import load_dotenv

load_dotenv("./.env")

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    """Get embedding from Google Gemini"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values