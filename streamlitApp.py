
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from src.helper import download_hf_embeddings, text_split, download_hf_model
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import os
from dotenv import load_dotenv
from src.prompt import prompt_template
from langchain.chains import RetrievalQA
import time
from pinecone import Pinecone
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
index_name = "medicure-chatbot"

# Set page configuration
st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #333;
        transform: scale(1.05);
        color:#fff;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f8ff ;
        color: #333;
        text-align: center;
        
    }
    .social-icons a {
        color: #333;
        margin: 0 10px;
        font-size: 24px;
    }
    .social-icons a>social-icons a:hover {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🏥 Medicure RAG Chatbot")

# Display welcome message
st.write("Welcome to Medicure Chatbot! Ask any medical question and I'll do my best to help you.")
st.write("#### Built with 🤗 Ctransformers, Langchain, and Pinecone VectorDB. Powered by Metal-llama2-7b-chat quantized LLM")
st.write("##### Resource Used 📖 : The Gale Encyclopedia of Medicine ")

# Parameters section
st.sidebar.header("Parameters")
k_value = st.sidebar.slider("Number of relevant documents (k)", min_value=1, max_value=3, value=2)
max_new_tokens = st.sidebar.slider("Max new tokens", min_value=64, max_value=1024, value=512)
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

# Initialize the chatbot components
@st.cache_resource
def initialize_chatbot(k, max_tokens, temp):
    embeddings = download_hf_embeddings()
    model_path = "TheBloke/Llama-2-7B-Chat-GGML"
    llm = CTransformers(model=model_path,
                        model_type="llama",
                        config={'max_new_tokens': max_tokens,
                                'temperature': temp})
    
    # initialize pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    docsearch = LangchainPinecone(index, embeddings.embed_query, "text")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': k}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)
    return qa

qa = initialize_chatbot(k_value, max_new_tokens, temperature)

# Chat interface
user_input = st.text_input("Ask your question:")
if st.button("Send", key="send"):

    if user_input:
        with st.spinner("Thinking..."):
            result = qa({"query": user_input})
            response = result["result"]
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# Display chat history
st.subheader("Chat History")
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")


# Animated loading for visual appeal
def load_animation():
    with st.empty():
        for i in range(3):
            for j in ["⋅", "⋅⋅", "⋅⋅⋅", "⋅⋅⋅⋅"]:
                st.write(f"Loading{j}")
                time.sleep(0.2)
            st.write("")


# Footer with social links
st.markdown("""
<div class="footer">
    <div class="social-icons">
        <a href="https://github.com/4darsh-Dev" target="_blank"><i class="fab fa-github"></i></a>
        <a href="https://linkedin.com/in/adarsh-maurya-dev" target="_blank"><i class="fab fa-linkedin"></i></a>
        <a href="https://adarshmaurya.onionreads.com" target="_blank"><i class="fas fa-globe"></i></a>
        <a href="https://www.kaggle.com/adarshm09" target="_blank"><i class="fab fa-kaggle"></i></a>
    </div>
    <p> <p style="text-align:center;">Made with ❤️ by <a href="https://www.adarshmaurya.onionreads.com">Adarsh Maurya</a></p> </p>
</div>
""", unsafe_allow_html=True)

# Load Font Awesome for icons
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">', unsafe_allow_html=True)