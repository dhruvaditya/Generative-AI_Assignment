from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
import google.ai.generativelanguage as glm
# from langchain_palm import PalmEmbeddings  # Import Palm Embeddings
from tempfile import NamedTemporaryFile
from typing import List
from chromadb.utils import embedding_functions
#https://docs.trychroma.com/embeddings Custom Embedding function
from chromadb import Documents, EmbeddingFunction, Embeddings

# class MyEmbeddingFunction(EmbeddingFunction):
#     def __call__(self, input: Documents) -> Embeddings:
#         # embed the documents somehow
#         return embeddings
#     #from chromadb.utils import embedding_functions given this function call
# default_ef = embedding_functions.DefaultEmbeddingFunction()
app = FastAPI()

# Initialize Chroma DB
db = Chroma()
# def get_api_key():
#     # Assuming you have a function to retrieve the API key from somewhere
#     # Replace this with your actual method of retrieving the API key
#     return "AIzaSyCCL7n1ueOC8zxOX_LMn0tWPdqMk7iOAdM"

# Function to configure GenAI with the API key
# def configure_genai(api_key: str = Depends(get_api_key)):
#     genai.configure(api_key=api_key)
#     return genai
#         #doc_search_emb.ipynb || From google embedding technique
# title = "The next generation of AI for developers and Google Workspace"
# sample_text = ("Title: The next generation of AI for developers and Google Workspace"
#     "\n"
#     "Full article:\n"
#     "\n"
#     "Gemini API & Google AI Studio: An approachable way to explore and prototype with generative AI applications")

# model = 'models/embedding-001'
# embedding = genai.embed_content(model=model,
#                                 content=sample_text,
#                                 task_type="retrieval_document",
#                                 title=title)

# print(embedding)

# @app.get('/embedded result')
# async def get_information(embedding):
#     return embedding


@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        if file.content_type == "application/pdf":
            # Read PDF file content
            contents = await file.read()
            # Extract text from PDF
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(contents)
                tmp_file_name = tmp_file.name
            text_loader = TextLoader(tmp_file_name)
            raw_documents = text_loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(raw_documents)
            # Embed text into vectors
            embeddings = PalmEmbeddings(api_key="AIzaSyCCL7n1ueOC8zxOX_LMn0tWPdqMk7iOAdM")
            vector_documents = embeddings.embed_documents(documents)
            # Store vectors in Chroma DB
            for vector_document in vector_documents:
                db.add_document(vector_document)
            # Clean up temporary file
            tmp_file.close()
            # Return success message
            return {"message": "PDF file uploaded and processed successfully"}
        else:
            return {"error": "Only PDF files are allowed."}
        
