# Q&A chatbot along with text summarizer

import streamlit as st
import os
from langchain_cohere import ChatCohere
from diffusers import StableDiffusionPipeline
import torch

# ---------- Cohere (Q&A) ----------
def get_text_response(question: str) -> str:
    llm = ChatCohere(
        model="command-r-plus",
        temperature=0.5,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    resp = llm.invoke(question)
    return resp.content
# ---------- Streamlit UI ----------
st.set_page_config(page_title="Cohere Q&A + Stable Diffusion")

st.title("🤖 Cohere Q&A + 🎨 Stable Diffusion (Text-to-Image)")

mode = st.radio("Choose mode:", ("Q&A (Cohere)", "Text-to-Image (Stable Diffusion)"))
user_input = st.text_input("Enter your question or prompt:")

if st.button("Submit") and user_input:
    if mode == "Q&A (Cohere)":
        answer = get_text_response(user_input)
        st.subheader("Answer")
        st.write(answer)


