# # Q&A Chatbot
# from langchain.llms import OpenAI

# #from dotenv import load_dotenv

# #load_dotenv()  # take environment variables from .env.

# import streamlit as st
# import os


# ## Function to load OpenAI model and get respones

# def get_openai_response(question):
#     llm=OpenAI(model_name="text-davinci-003",temperature=0.5)
#     response=llm(question)
#     return response

# ##initialize our streamlit app

# st.set_page_config(page_title="Q&A Demo")

# st.header("Langchain Application")

# input=st.text_input("Input: ",key="input")
# response=get_openai_response(input)

# submit=st.button("Ask the question")

# ## If ask button is clicked

# if submit:
#     st.subheader("The Response is")
#     st.write(response)



# import streamlit as st
# import os
# from langchain_cohere import ChatCohere   # ✅ new import

# ## Function to load Cohere model and get responses
# def get_cohere_response(question):
#     llm = ChatCohere(
#         model="command-r",   # ✅ supported chat model
#         temperature=0.5,
#         cohere_api_key=os.getenv("COHERE_API_KEY")
#     )
#     response = llm.invoke(question)   # ✅ use invoke, not call
#     return response.content           # ChatCohere returns a ChatMessage

# ## initialize our streamlit app
# st.set_page_config(page_title="Q&A Demo")

# st.header("Langchain + Cohere Application")

# input_text = st.text_input("Input: ", key="input")

# submit = st.button("Ask the question")

# ## If ask button is clicked
# if submit and input_text:
#     response = get_cohere_response(input_text)
#     st.subheader("The Response is")
#     st.write(response)

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

