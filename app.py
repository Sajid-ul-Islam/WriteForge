import os
import datetime
import streamlit as st
from dotenv import load_dotenv
from ai_providers import get_provider
from prompts import SYSTEM_PROMPT, TONES, LENGTHS, build_article_prompt, build_headlines_prompt

load_dotenv()

st.set_page_config(page_title="WriteForge AI", page_icon="✍️", layout="centered")

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { max-width: 900px; margin: 0 auto; }
    .output-box { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

st.title("WriteForge AI")
st.caption("Convert raw thoughts in any language into publish-ready English articles")

# --- Sidebar: Provider Config ---
with st.sidebar:
    st.header("AI Provider")
    provider_name = st.selectbox("Provider", ["OpenAI", "Gemini", "Groq", "HuggingFace", "OpenRouter", "Ollama"])

    if provider_name == "OpenAI":
        default_key = st.secrets.get("llm", {}).get("openai_key", os.getenv("OPENAI_API_KEY", ""))
        api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"])
        provider_kwargs = {"api_key": api_key, "model": model}
    elif provider_name == "Gemini":
        default_key = st.secrets.get("llm", {}).get("gemini_key", os.getenv("GEMINI_API_KEY", ""))
        api_key = st.text_input("Gemini API Key", value=default_key, type="password")
        model = st.selectbox("Model", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])
        provider_kwargs = {"api_key": api_key, "model": model}
    elif provider_name == "Groq":
        default_key = st.secrets.get("llm", {}).get("groq_key", os.getenv("GROQ_API_KEY", ""))
        api_key = st.text_input("Groq API Key", value=default_key, type="password")
        model = st.selectbox("Model", ["llama3-70b-8192", "mixtral-8x7b-32768"])
        provider_kwargs = {"api_key": api_key, "model": model}
    elif provider_name == "HuggingFace":
        default_key = st.secrets.get("llm", {}).get("huggingface_key", os.getenv("HF_API_KEY", ""))
        api_key = st.text_input("HF API Key", value=default_key, type="password")
        model = st.text_input("Model ID", value="mistralai/Mistral-7B-Instruct-v0.2")
        provider_kwargs = {"api_key": api_key, "model": model}
    elif provider_name == "OpenRouter":
        default_key = st.secrets.get("llm", {}).get("openrouter_key", os.getenv("OPENROUTER_API_KEY", ""))
        api_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
        model = st.text_input("Model ID", value="google/gemini-2.0-flash-001")
        provider_kwargs = {"api_key": api_key, "model": model}
    else:
        base_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        model = st.text_input("Model Name", value=os.getenv("OLLAMA_MODEL", "llama3"))
        provider_kwargs = {"base_url": base_url, "model": model}

    st.divider()
    st.header("Settings")
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)

def validate_config():
    if provider_name != "Ollama" and not provider_kwargs.get("api_key"):
        st.error(f"🔑 API Key missing for {provider_name}. Please add it to `.streamlit/secrets.toml` or enter it in the sidebar.")
        st.stop()

# --- Main Input ---
user_input = st.text_area("Paste your text here (any language):", height=200, placeholder="Enter your raw text, notes, or ideas...")

col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox("Tone", TONES)
with col2:
    length = st.selectbox("Length", list(LENGTHS.keys()))

col_gen, col_head = st.columns(2)
generate_btn = col_gen.button("Generate Article", type="primary", use_container_width=True)
headlines_btn = col_head.button("Generate Headlines", use_container_width=True)

# --- Generate Article ---
if generate_btn:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        validate_config()
        
        try:
            provider = get_provider(provider_name, **provider_kwargs)
        except Exception as e:
            st.error(f"Provider error: {e}")
            st.stop()

        with st.spinner(f"Generating with {provider.name()}..."):
            try:
                prompt = build_article_prompt(user_input, tone, length)
                article = provider.generate(prompt, SYSTEM_PROMPT, temperature)

                st.success("Article generated!")
                st.markdown("---")
                st.markdown(article)

                # Save to History
                st.session_state.history.append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "type": "Article",
                    "tone": tone,
                    "content": article,
                    "provider": provider.name(),
                    "params": {
                        "provider_name": provider_name,
                        "provider_kwargs": provider_kwargs.copy(),
                        "user_input": user_input,
                        "tone": tone,
                        "length": length,
                        "temperature": temperature
                    }
                })

                # Analysis
                try:
                    from textblob import TextBlob
                    blob = TextBlob(article)
                    col_s, col_sub, col_wc = st.columns(3)
                    col_s.metric("Sentiment", f"{blob.sentiment.polarity:.2f}")
                    col_sub.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
                    col_wc.metric("Word Count", len(article.split()))
                except ImportError:
                    pass

                st.download_button(
                    "Download Article",
                    data=article,
                    file_name="article.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Generation failed: {e}")

# --- Generate Headlines ---
if headlines_btn:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        validate_config()

        try:
            provider = get_provider(provider_name, **provider_kwargs)
        except Exception as e:
            st.error(f"Provider error: {e}")
            st.stop()

        with st.spinner("Generating headlines..."):
            try:
                prompt = build_headlines_prompt(user_input, tone)
                headlines = provider.generate(prompt, SYSTEM_PROMPT, temperature)
                
                # Save to History
                st.session_state.history.append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "type": "Headlines",
                    "tone": tone,
                    "content": headlines,
                    "provider": provider.name(),
                    "params": {
                        "provider_name": provider_name,
                        "provider_kwargs": provider_kwargs.copy(),
                        "user_input": user_input,
                        "tone": tone,
                        "temperature": temperature
                    }
                })
                
                st.markdown("### Headline Options")
                st.markdown(headlines)
            except Exception as e:
                st.error(f"Failed: {e}")

# --- History Section ---
if st.session_state.history:
    st.divider()
    
    col_h, col_c = st.columns([3, 1])
    col_h.header("Session History")
    if col_c.button("Clear History", use_container_width=True, help="Remove all items from session history"):
        st.session_state.history = []
        st.rerun()

    for idx, item in enumerate(reversed(st.session_state.history)):
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
                with st.spinner("Retrying generation..."):
                    try:
                        r_prov = get_provider(params["provider_name"], **params["provider_kwargs"])
                        if item["type"] == "Article":
                            r_prompt = build_article_prompt(params["user_input"], params["tone"], params["length"])
                        else:
                            r_prompt = build_headlines_prompt(params["user_input"], params["tone"])
                        
                        r_content = r_prov.generate(r_prompt, SYSTEM_PROMPT, params["temperature"])
                        
                        # Save new generation to history
                        st.session_state.history.append({
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "type": item["type"],
                            "tone": params["tone"],
                            "content": r_content,
                            "provider": r_prov.name(),
                            "params": params
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retry failed: {e}")
