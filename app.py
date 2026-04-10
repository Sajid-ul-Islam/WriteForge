import os
import streamlit as st
from dotenv import load_dotenv
from ai_providers import get_provider
from prompts import SYSTEM_PROMPT, TONES, LENGTHS, build_article_prompt, build_headlines_prompt

load_dotenv()

st.set_page_config(page_title="WriteForge AI", page_icon="✍️", layout="centered")

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
    provider_name = st.selectbox("Provider", ["OpenAI", "Gemini", "Ollama"])

    if provider_name == "OpenAI":
        api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"])
        provider_kwargs = {"api_key": api_key, "model": model}
    elif provider_name == "Gemini":
        api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
        model = st.selectbox("Model", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])
        provider_kwargs = {"api_key": api_key, "model": model}
    else:
        base_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        model = st.text_input("Model Name", value=os.getenv("OLLAMA_MODEL", "llama3"))
        provider_kwargs = {"base_url": base_url, "model": model}

    st.divider()
    st.header("Settings")
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)

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
        try:
            provider = get_provider(provider_name, **provider_kwargs)
        except Exception as e:
            st.error(f"Provider error: {e}")
            st.stop()

        with st.spinner("Generating headlines..."):
            try:
                prompt = build_headlines_prompt(user_input, tone)
                headlines = provider.generate(prompt, SYSTEM_PROMPT, temperature)
                st.markdown("### Headline Options")
                st.markdown(headlines)
            except Exception as e:
                st.error(f"Failed: {e}")
