from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain import Chroma, SentenceTransformerEmbeddings 
from typing import List

app = FastAPI()

# @app.post("/upload/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     uploaded_files = []
#     for file in files:
#         if file.content_type == "application/pdf":
#             contents = await file.read()
#             uploaded_files.append({"filename": file.filename, "content": contents.decode()})
#         else:
#             return JSONResponse(status_code=400, content={"error": "Only PDF files are allowed."})
#     return JSONResponse(status_code=200, content={"message": "Files uploaded successfully", "files": uploaded_files})
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            if file.content_type == "application/pdf":
                contents = await file.read()
                uploaded_files.append({"filename": file.filename, "content": contents.decode()})
            else:
                return JSONResponse(status_code=400, content={"error": "Only PDF files are allowed."})
        return JSONResponse(status_code=200, content={"message": "Files uploaded successfully", "files": uploaded_files})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})




# Get data from POST 
data = request.json()  

# Create embedding function
embeddings = SentenceTransformerEmbeddings()

# Create Chroma instance  
db = Chroma()  

# Add embedded data
db.add_texts(texts=data, embeddings=embeddings) 