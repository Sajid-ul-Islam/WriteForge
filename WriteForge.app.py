import os
import datetime
import sys
import importlib.metadata
import json
import io
import zipfile
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from ai_providers import get_provider
from prompts import SYSTEM_PROMPT, TONES, LENGTHS, build_article_prompt, build_headlines_prompt, build_poem_prompt

load_dotenv()

# --- Logging Configuration ---
# Tracks detailed AI provider errors (Quota, Auth, Connectivity) to 'provider_errors.log'.
logging.basicConfig(
    filename="provider_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

HISTORY_FILE = "history.json"
VECTOR_DB_DIR = "vector_store"

def load_history():
    """Loads history from a local JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
    return []

@st.cache_resource
def get_embeddings_model():
    """Initializes and caches the HuggingFace embedding model."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Embedding model initialization failed: {e}")
        return None

def load_vector_db():
    """Loads the FAISS vector store from local disk."""
    if os.path.exists(VECTOR_DB_DIR):
        try:
            from langchain_community.vectorstores import FAISS
            embeddings = get_embeddings_model()
            if not embeddings:
                return None
            return FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Failed to load vector DB: {e}")
    return None

def vectorize_text(text):
    """Chunks text and appends to or creates a local FAISS vector store."""
    with st.status("Updating Knowledge Base...", expanded=True) as status:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import FAISS
            
            status.write("Splitting document into manageable chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(text)
            
            status.write("Loading embedding model (this may take a moment)...")
            # Using a lightweight local embedding model
            embeddings = get_embeddings_model()
            if not embeddings:
                raise ImportError("langchain-huggingface or sentence-transformers missing.")

            if os.path.exists(VECTOR_DB_DIR):
                status.write(f"Appending to existing vector store in {VECTOR_DB_DIR}...")
                vector_db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
                vector_db.add_texts(chunks)
            else:
                status.write("Creating new vector index...")
                vector_db = FAISS.from_texts(chunks, embeddings)

            status.write("Persisting index to disk...")
            vector_db.save_local(VECTOR_DB_DIR)
            
            status.update(label="Knowledge Base Updated!", state="complete", expanded=False)
            return vector_db
        except Exception as e:
            status.update(label="Vectorization Failed", state="error")
            logger.error(f"Vectorization failed: {e}")
            st.error("Could not update knowledge base. Check logs for details.")
            return None

def save_history(history):
    """Saves history to a local JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

st.set_page_config(page_title="WriteForge AI", page_icon="✍️", layout="centered")

# --- Dependency Check ---
def check_dependencies():
    """Verifies that core packages from requirements.txt are installed."""
    core_packages = [
        "streamlit", "openai", "google-generativeai", "anthropic", "groq",
        "langchain", "langchain-community", "langchain-huggingface", 
        "faiss-cpu", "sentence-transformers", "textblob", "pypdf", "python-docx"
    ]
    missing = []
    for pkg in core_packages:
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)

    return missing

missing_deps = check_dependencies()

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_vector_db()
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { max-width: 900px; margin: 0 auto; }
    .output-box { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

st.title("WriteForge AI")
st.caption("Convert raw thoughts in any language into publish-ready English articles")

# --- Sidebar: Settings ---
with st.sidebar:
    if missing_deps:
        with st.expander("⚠️ System Warnings", expanded=False):
            st.warning("Some features are disabled due to missing packages:")
            st.write(f"`{', '.join(missing_deps)}`")
            st.info("To enable all features, run:")
            st.code("pip install -r requirements.txt")

    st.header("Settings")
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)
    rag_enabled = st.checkbox("Enable RAG (Context Enhancement)", value=True, help="Automatically use uploaded documents to enhance article generation.")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        import shutil
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR)
        st.session_state.vector_db = None
        st.session_state.last_uploaded_file = None
        st.toast("Knowledge base cleared!", icon="🗑️")
        st.rerun()

    st.divider()
    st.header("Diagnostics")
    if st.button("🔍 Run Health Check", use_container_width=True):
        candidates = get_provider_candidates()
        for p_name, p_kwargs in candidates:
            try:
                provider = get_provider(p_name, **p_kwargs)
                if provider.validate():
                    st.sidebar.success(f"✅ {p_name}: Ready")
            except Exception as e:
                st.sidebar.error(f"❌ {p_name}: {str(e)}")

    st.divider()
    st.header("Provider Logs")
    auto_refresh = st.checkbox("Auto-refresh logs (10s)", help="Automatically refresh the app to see new log entries")
    
    if os.path.exists("provider_errors.log"):
        with open("provider_errors.log", "r", encoding="utf-8") as f:
            log_content = f.read()
        st.text_area("Recent Errors", value=log_content, height=300, disabled=True)
        if st.button("Clear Log File", use_container_width=True):
            open("provider_errors.log", "w").close()
            st.rerun()
    else:
        st.info("No provider logs recorded.")

# --- AI Fallback Logic ---
# Secondary models to try if the primary one is decommissioned or not found.
MODEL_FALLBACKS = {
    "Anthropic": ["claude-3-5-haiku-20241022", "claude-3-haiku-20240307"],
    "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "OpenAI": ["gpt-4o", "gpt-4-turbo"],
    "Groq": ["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "OpenRouter": ["anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-405b-instruct"],
}

def is_model_error(error: Exception) -> bool:
    """
    Detects if an error is model-specific (e.g., 400 decommissioned or 404 not found),
    indicating that a different model ID might work for the same provider.
    """
    msg = str(error).lower()
    return any(keyword in msg for keyword in ["model_not_found", "decommissioned", "404", "no endpoints found"])


def run_generation_flow(task_type: str, text: str, tone: str, length: str = None, rag_enabled: bool = True):
    """
    Handles the core logic for AI generation including candidate selection,
    model fallbacks, error logging, and history management.
    """
    candidates = get_provider_candidates()
    result = None
    
    for p_name, p_kwargs in candidates:
        # Build list of models to try for this provider
        models_to_try = [p_kwargs.get("model")] if "model" in p_kwargs else []
        if p_name in MODEL_FALLBACKS:
            models_to_try.extend([m for m in MODEL_FALLBACKS[p_name] if m != models_to_try[0]])
        
        provider_success = False
        for current_model in models_to_try:
            with st.spinner(f"Generating {task_type.lower()} with {p_name} ({current_model})..."):
                try:
                    # Update configuration for this specific model attempt
                    current_kwargs = p_kwargs.copy()
                    if "model" in current_kwargs:
                        current_kwargs["model"] = current_model
                    
                    provider = get_provider(p_name, **current_kwargs)
                    
                    if task_type == "Article":
                        # Automatic RAG: Fetch relevant context if vector_db is present and enabled
                        context = ""
                        if rag_enabled and st.session_state.vector_db:
                            with st.spinner("Retrieving context from knowledge base..."):
                                docs = st.session_state.vector_db.similarity_search(text, k=3)
                                context = "\n\n".join([d.page_content for d in docs])
                                if context:
                                    st.info("💡 RAG: Context successfully retrieved from document.")

                        prompt = build_article_prompt(text, tone, length, context=context)
                    elif task_type == "Headlines":
                        prompt = build_headlines_prompt(text, tone)
                    else: # Poem
                        prompt = build_poem_prompt(text, tone)
                        
                    content = provider.generate(prompt, SYSTEM_PROMPT, temperature)
                    
                    # Save to History
                    history_params = {
                        "provider_name": p_name,
                        "provider_kwargs": current_kwargs,
                        "user_input": text,
                        "tone": tone,
                        "temperature": temperature
                    }
                    if task_type == "Article":
                        history_params["length"] = length
                        
                    st.session_state.history.append({
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "type": task_type,
                        "tone": tone,
                        "content": content,
                        "provider": provider.name(),
                        "params": history_params
                    })
                    save_history(st.session_state.history)

                    # Display results
                    if task_type in ["Article", "Poem"]:
                        st.success(f"{task_type} generated via {provider.name()}!")
                        st.markdown("---")
                        st.markdown(content)
                        st.caption(f"🚀 Model: {provider.name()}")

                        # Analysis
                        try:
                            from textblob import TextBlob
                            import nltk
                            # Ensure required corpora are available for Streamlit Cloud
                            try:
                                nltk.data.find('tokenizers/punkt')
                            except (LookupError, AttributeError):
                                nltk.download('punkt')
                                nltk.download('brown')
                            blob = TextBlob(content)
                            col_s, col_sub, col_wc = st.columns(3)
                            col_s.metric("Sentiment", f"{blob.sentiment.polarity:.2f}")
                            col_sub.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
                            col_wc.metric("Word Count", len(content.split()))
                        except ImportError: pass

                        st.download_button(
                            f"Download {task_type}", 
                            data=content, 
                            file_name=f"{task_type.lower()}.md", 
                            mime="text/markdown"
                        )
                    else:
                        st.markdown("### Headline Options")
                        st.markdown(content)
                        st.caption(f"🚀 Model: {provider.name()}")
                    
                    result = content
                    provider_success = True
                    break 
                except Exception as e:
                    if is_model_error(e) and current_model != models_to_try[-1]:
                        logger.warning(f"{task_type} model {current_model} failed for {p_name}, trying fallback: {e}")
                        continue 
                    
                    logger.error(f"{task_type} generation failed for {p_name} ({current_model}): {str(e)}")
                    break 
        
        if provider_success:
            break
            
    if not result:
        st.error(f"All available AI providers failed to generate the {task_type.lower()}.")
    return result


def get_provider_candidates():
    """Determine which providers are configured in secrets/env."""
    candidates = []
    llm_sec = st.secrets.get("llm", {})
    
    # 1. Anthropic
    a_key = llm_sec.get("anthropic_key") or os.getenv("ANTHROPIC_API_KEY")
    if a_key:
        candidates.append(("Anthropic", {"api_key": a_key, "model": "claude-3-5-sonnet-20240620"}))

    # 2. Gemini
    g_key = llm_sec.get("gemini_key") or os.getenv("GEMINI_API_KEY")
    if g_key:
        candidates.append(("Gemini", {"api_key": g_key, "model": "gemini-2.0-flash"}))
    
    # 3. OpenAI
    o_key = llm_sec.get("openai_key") or os.getenv("OPENAI_API_KEY")
    if o_key:
        candidates.append(("OpenAI", {"api_key": o_key, "model": "gpt-4o-mini"}))
        
    # 4. Groq
    gr_key = llm_sec.get("groq_key") or os.getenv("GROQ_API_KEY")
    if gr_key:
        candidates.append(("Groq", {"api_key": gr_key, "model": "llama-3.3-70b-versatile"}))

    # 5. OpenRouter
    or_key = llm_sec.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY")
    if or_key:
        candidates.append(("OpenRouter", {"api_key": or_key, "model": "google/gemini-2.0-flash-001"}))

    # 6. Ollama (Local fallback)
    candidates.append(("Ollama", {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), 
        "model": os.getenv("OLLAMA_MODEL", "llama3")
    }))
    
    return candidates

# --- Main Input ---
uploaded_file = st.file_uploader("📄 Upload a local file (.txt, .md, .pdf, .docx)", type=["txt", "md", "pdf", "docx"], help="Select a file from your machine to auto-detect its content.")

initial_text = ""
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".pdf"):
            import pypdf
            pdf_reader = pypdf.PdfReader(uploaded_file)
            initial_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif uploaded_file.name.endswith(".docx"):
            import docx
            doc = docx.Document(uploaded_file)
            initial_text = "\n".join([para.text for para in doc.paragraphs])
        else:
            initial_text = uploaded_file.read().decode("utf-8")
        
        # Automatically vectorize the content if it's a new file
        if uploaded_file.name != st.session_state.last_uploaded_file:
            st.session_state.vector_db = vectorize_text(initial_text)
            st.session_state.last_uploaded_file = uploaded_file.name
            st.toast("Document successfully vectorized!", icon="✅")
                
    except Exception as e:
        st.error(f"Error reading local file: {e}")

if st.session_state.vector_db:
    with st.expander("🔍 Semantic Search (Query Document)"):
        query = st.text_input("Ask the document a question to extract context:")
        if query:
            docs = st.session_state.vector_db.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            st.markdown("**Relevant Context Found:**")
            st.info(context)
            if st.button("Use this context"):
                initial_text = context

user_input = st.text_area("Paste your text here (any language):", value=initial_text, height=200, placeholder="Enter your raw text, notes, or ideas...")

if not get_provider_candidates():
    st.error("No AI providers configured. Please add API keys to `.streamlit/secrets.toml`.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox("Tone", TONES)
with col2:
    length = st.selectbox("Length", list(LENGTHS.keys()))

col_gen, col_head, col_poem = st.columns(3)
generate_btn = col_gen.button("Generate Article", type="primary", use_container_width=True)
headlines_btn = col_head.button("Generate Headlines", use_container_width=True)
poem_btn = col_poem.button("Generate Poem", use_container_width=True)

# --- Generate Article ---
if generate_btn:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        run_generation_flow("Article", user_input, tone, length, rag_enabled=rag_enabled)

# --- Generate Headlines ---
if headlines_btn:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        run_generation_flow("Headlines", user_input, tone, rag_enabled=rag_enabled)

# --- Generate Poem ---
if poem_btn:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        run_generation_flow("Poem", user_input, tone, rag_enabled=rag_enabled)

# --- History Section ---
if st.session_state.history:
    st.divider()
    
    col_h, col_c = st.columns([3, 1])
    col_h.header("Session History")
    if col_c.button("Clear History", use_container_width=True, help="Remove all items from session history"):
        st.session_state.history = []
        save_history([])
        st.rerun()

    # --- Search & Filter UI ---
    s_col1, s_col2 = st.columns([2, 1])
    search_query = s_col1.text_input("🔍 Search Keyword", placeholder="Search in content or headlines...")
    selected_tones = s_col2.multiselect("Filter Tone", TONES)

    # Filtering logic
    filtered_history = [
        item for item in st.session_state.history
        if (not search_query or search_query.lower() in item['content'].lower()) and
           (not selected_tones or item['tone'] in selected_tones)
    ]

    if not filtered_history:
        st.info("No results match your filters.")
    else:
        # --- ZIP Export Logic ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, item in enumerate(filtered_history):
                # Construct a descriptive filename for each markdown file
                file_time = item['timestamp'].replace(":", "-")
                file_type = item['type'].lower()
                filename = f"{file_time}_{file_type}_{i+1}.md"
                zf.writestr(filename, item['content'])
        
        st.download_button(
            label="📦 Export Filtered Results to ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"writeforge_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
            help="Download all currently visible (filtered) history items in a single ZIP file."
        )

    for idx, item in enumerate(reversed(filtered_history)):
        with st.expander(f"{item['timestamp']} - {item['type']} ({item['tone']})"):
            st.caption(f"Generated via {item['provider']}")
            st.markdown(item['content'])
            
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download Result",
                data=item['content'],
                file_name=f"history_{idx}.md",
                key=f"dl_{idx}",
                use_container_width=True
            )
            
            params = item.get("params")
            if params and c2.button("🔄 Retry", key=f"retry_{idx}", use_container_width=True):
                run_generation_flow(
                    item["type"], 
                    params["user_input"], 
                    params["tone"], 
                    params.get("length"),
                    rag_enabled=rag_enabled
                )
                st.rerun()

# --- Auto-refresh Logic ---
if auto_refresh:
    time.sleep(10)
    st.rerun()
