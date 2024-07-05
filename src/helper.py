from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download


# loading the data
def load_data(path):
    loader = PyPDFDirectoryLoader(path)
    extracted_data = loader.load()
    return extracted_data


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


#download embedding model
def download_hf_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


# downloading any pdf on web

import os
import requests

def download_pdf(url):
    if not os.path.exists('data'):
        os.makedirs('data')

    pdf_url = url

    # Get the filename from the URL
    filename = pdf_url.split("/")[-1]

    # Full path where the PDF will be saved
    save_path = os.path.join('data', filename)

    # Download the PDF
    response = requests.get(pdf_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a file
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded and saved to {save_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")





def download_hf_model(model_name_or_path, model_basename):
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    return model_path

