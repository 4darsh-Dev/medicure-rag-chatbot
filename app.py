

from langchain_community.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from src.helper import download_hf_embeddings,text_split,download_hf_model
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import pinecone
import os
from dotenv import load_dotenv
from src.prompt import prompt_template
from langchain_community.chains import RetrievalQA

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# loading embeddigns model
embeddings = download_hf_embeddings()

## intialize the pinecone client


# Loading the LLM

##  hugging face quantized model (llama2-7b-chat)
model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin" 

model_path = download_hf_model(model_name_or_path, model_basename)
llm=CTransformers(model=model_path,
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


## prompt prepartion

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}


# Create a LangChain vectorstore
index_name="medicure-chatbot"
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(index_name)

docsearch = LangchainPinecone(index, embeddings.embed_query, "text")

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


while True:
    user_input=input(f"Input Prompt Stop(s):")
    if user_input == "s":
        break
    result=qa({"query": user_input})
    print("Response : ", result["result"])
