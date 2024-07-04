from src.helper import load_data, text_split, download_hf_embeddings

embeddings = download_hf_embeddings()
query = input("Enter your query: ")

# Generate embedding for the query
query_embedding = embeddings.embed_query(query)

# Perform similarity search
search_results = index.query(
    namespace="ns1",  # Replace with your actual namespace
    vector=query_embedding,
    top_k=1,  # Number of results you want
    include_values=True,
    include_metadata=True
)

# Process and print results
print("Results:")
for match in search_results['matches']:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")
    print("---")