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

# ---------- Stable Diffusion (Text-to-Image) ----------
@st.cache_resource
def load_diffusion_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

def generate_image(prompt: str):
    pipe = load_diffusion_pipeline()
    image = pipe(prompt).images[0]
    return image

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

    elif mode == "Text-to-Image (Stable Diffusion)":
        with st.spinner("Generating image..."):
            image = generate_image(user_input)
            st.subheader("Generated Image")
            st.image(image, caption=user_input)


