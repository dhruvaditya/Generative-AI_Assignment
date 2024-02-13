from fastapi import FastAPI, UploadFile, File
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_palm import PalmEmbeddings  # Import Palm Embeddings
from tempfile import NamedTemporaryFile
from typing import List

app = FastAPI()

# Initialize Chroma DB
db = Chroma()

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
