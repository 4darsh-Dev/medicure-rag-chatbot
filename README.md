# 💡Medicure RAG Chatbot🤖
An AI chatbot implementing RAG technique with Meta-Llama2-7b Large Language Model, along with langchain and pinecone vector databases.
Resource Used 📖 : The Gale Encyclopedia of Medicine


## Screenshots

<p align="center">
  <img src="https://onionreads.com/wp-content/uploads/2024/07/Screenshot-2024-07-05-171202.png" alt="Medicure adarsh maurya onionreads " width="600px" />
</p>


## Technologies Used
1. Streamlit- WebApp UI
2. Pinecone - Vector Database
3. Langchain and sentence-transformers - RetrieveQAChain and Embedding Model
4. Meta Llama-2-7b-chat quantized Model - Large Language Model(LLM) from Hugging Face Hub


## Solution Approach
Pinecone vector db stores the text_chunks embeddings generated from the Book Pdf. LangChain is used for building the LLMChain with promptTemplate to perform the similarity search from pinecone and then fine-grain the output with LLM. 


## Running Web App Locally

To run  web app locally, follow these steps:

1.**Clone the Repo**:
  ```bash
  git clone https://github.com/4darsh-Dev/medicure-rag-chatbot.git 
  ```


2. **Configure poetry**:
    ```bash
      pip install poetry
      poetry init
      poetry shell
      
    ```
3. **Install Requirements**: 

    ```bash
      poetry install
      
    ```


4. **Run the Streamlit App**:

    ```bash
    poetry streamlit run app.py
    ```

5. **Access Your App**: After running the command, Streamlit will start a local web server and provide a URL where you can access your app. Typically, it will be something like `http://localhost:8501`. Open this URL in your web browser.

6. **Stop the Streamlit Server**: To stop the Streamlit server, go back to the terminal or command prompt where it's running and press `Ctrl + C` to terminate the server.


# Hi, I'm Adarsh! 👋


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://adarshmaurya.onionreads.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adarsh-maurya-dev/)


## Feedback

If you have any feedback, please reach out to us at adarsh@onionreads.com
