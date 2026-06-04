# Agent Architecture

The WriteForge agent operates as a **Retrieval-Augmented Editorial Agent**.

## Persona
The agent is defined as an "Elite Editorial Writer." It prioritizes clarity, transition smoothness, and the removal of "filler" or "fluff."

## Workflow Logic
1. **Ingestion**: The agent receives raw text or an uploaded file.
2. **Knowledge Retrieval**: If RAG is enabled, the agent performs a similarity search against the local FAISS index using the input as a query.
3. **Context Augmentation**: Relevant snippets are injected into the prompt as "Reference Context."
4. **Model Orchestration**: The system iterates through configured AI providers. If a 429 (Rate Limit) or 400 (Model Error) is encountered, it triggers the fallback logic to the next best model.
5. **Post-Processing**: The generated markdown is analyzed for word count and sentiment before being displayed.

## Fallback Priority
Anthropic -> Gemini -> OpenAI -> Groq -> OpenRouter -> Ollama (Local)