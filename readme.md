# WriteForge AI

WriteForge AI is a publish-ready content generation platform that converts raw thoughts into polished English articles, headlines, and poems. It features a robust multi-provider fallback system and a local RAG (Retrieval-Augmented Generation) knowledge base.

## Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Gemini, Anthropic, Groq, and OpenRouter.
- **Smart Fallback**: Automatically tries alternative models if the primary one fails due to quota or deprecation.
- **Local RAG**: Upload PDF, Docx, or Markdown files to build a persistent vector knowledge base using FAISS.
- **Persistent History**: Session data and generated content are saved locally for retrieval across sessions.
- **Export Options**: Export your filtered history as a ZIP archive of Markdown files.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file or use Streamlit secrets for your API keys.

## Usage

Run the application using Streamlit:
```bash
streamlit run app.py
```

## Configuration

Add your API keys to `.streamlit/secrets.toml`:
```toml
[llm]
openai_key = "your-key"
gemini_key = "your-key"
anthropic_key = "your-key"
groq_key = "your-key"
```