

import os
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv


from src.helper import load_data, text_split, download_hf_embeddings

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# text chunks from the pdf
extracted_data = load_data("data")

text_chunks = text_split(extracted_data)

# load the embeddings model
embeddings = download_hf_embeddings()


# Create embeddings for your text chunks
embedded_texts = embeddings.embed_documents([t.page_content for t in text_chunks])

index_name="medicure-chatbot"

# Prepare vectors for upsert
vectors_to_upsert = []
for i, (chunk, embedding) in enumerate(zip(text_chunks, embedded_texts)):
    vector = {
        "id": f"chunk_{i}",
        "values": embedding,
        "metadata": {
            "text": chunk.page_content,
            # Add any other metadata you want to include
        }
    }
    vectors_to_upsert.append(vector)

# Upsert vectors to Pinecone

# Function to split list into chunks
def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# Split vectors into smaller batches
batch_size = 100  
batches = chunk_list(vectors_to_upsert, batch_size)

# Upsert batches to Pinecone
for i, batch in enumerate(batches):
    try:
        index.upsert(
            vectors=batch,
            namespace="ns1"  # Replace with your desired namespace
        )
        print(f"Batch {i+1}/{len(batches)} upserted successfully")
    except Exception as e:
        print(f"Error upserting batch {i+1}: {str(e)}")
        # You might want to implement retry logic here

print("Upsert completed")