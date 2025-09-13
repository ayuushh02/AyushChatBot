# RAG Chatbot - Tech News Assistant

A simple RAG (Retrieval-Augmented Generation) chatbot using OpenAI, Pinecone, and Gradio.

## Setup

1. Install uv from https://docs.astral.sh/uv/
2. Create a `.env` file and add your API keys:
   ```bash
   OPENAI_API_KEY=YOUR_API_KEY
   OPENAI_BASE_URL=OPENAI_BASE_URL
   OPENAI_MODEL=PREFERRED_MODEL

   PINECONE_API_KEY=PINECONE_API_KEY
   PINECONE_INDEX=PINECONE_INDEX


   GEMINI_API_KEY=GEMINI_API_KEY
   ```
3. Install dependencies:

```bash
uv sync
```

4. Index the tech news data:

```bash
uv run python index.py
```

5. Run the chatbot:

```bash
uv run python main.py
```

The chatbot will open in your browser at http://localhost:7860

## What it does

- **index.py**: Loads text files from `source_text/` folder, splits them into chunks, and stores in Pinecone
- **main.py**: RAG chatbot that searches Pinecone for relevant context and answers questions about recent tech news
- **source_text/**: Contains 5 recent tech news articles about AI developments
