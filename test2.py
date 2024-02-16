# from fastapi import FastAPI, UploadFile, File,Form
# from secrets import token_hex
# import uvicorn

# app = FastAPI(title="Upload the pdf file")

# @app.post("/upload/")
# async def upload(file: UploadFile = File(...)):
#     file_ext = file.filename.split(".").pop()
#     file_name = token_hex(10)
#     file_path = f"{file_name}.{file_ext}"
#     with open(file_path, "wb") as f:
#         content = await file.read()
#         f.write(content)
#     return {"success": True, "file_path": file_path, "message": "File uploaded successfully"}
# @app.post("/process_pdf/")
# async def process_pdf(pdf_name: str = Form(...), question: str = Form(...)):
#     # Do something with the provided PDF name and question
#     return {"pdf_name": pdf_name, "question": question}



# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
import os
from fastapi import FastAPI, HTTPException
from PyPDF2 import PdfReader
# from chroma import Chroma
# from embeddings import GooglePalmEmbeddings
from typing import Optional

app = FastAPI()

# Initialize ChromaDB and embeddings
# embeddings = GooglePalmEmbeddings(model="models/embedding-001")
# chroma_db = Chroma()

@app.post("/store-pdf-in-chroma/")
async def store_pdf_in_chroma(pdf_path, pdf_name: Optional[str] = None):
    # pdf_path="C:/Users/promact.DESKTOP-RHBFB7T/Downloads/docu.pdf"
    # .venv\Scripts\activate.bat
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found.")
    
    try:
        # Read PDF and extract text
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Store text in ChromaDB
        # document_id = chroma_db.store_document(text, name=pdf_name)
        
        return {"message": f"PDF document '{pdf_name}' stored in ChromaDB with document ID ."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing PDF document in ChromaDB: {str(e)}")
