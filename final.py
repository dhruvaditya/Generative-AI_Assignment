from fastapi import FastAPI,UploadFile, Form, HTTPException
from pdfrw import PdfReader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
app = FastAPI()
@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = Form(...)):
    uploaded_documents = []
    extracted_texts = []
    embeddings = []

    for file in files:
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            return {"error": "Invalid file format. Please upload only PDFs."}

        try:
            # Read PDF using pdfrw
            pdf_reader = PdfReader(file.file)
            text = ""
            for page in pdf_reader.pages:
                text += page.ExtractText()

            extracted_texts.append(text)
        except Exception as e:
            return {"error": f"Error processing file {file.filename}: {e}"}

    for text in extracted_texts:
        # Split text into chunks
        chunks = text.split("\n\n")

        for chunk in chunks:
            # Encode chunks using pretrained model
            embeddings.append(sentence_transformer.encode(chunk))

    # Store embeddings in ChromaDB
    chroma_client.update(embeddings, "vector_collection")

    # Create Elasticsearch documents and upload
    documents = []
    for i, text in enumerate(extracted_texts):
        document = {
            "id": i,
            "text": text,
            "embedding": embeddings[i],
        }
        documents.append(document)
    es_client.index(index=ELASTICSEARCH_INDEX, body=documents)

    return {"message": "PDFs uploaded and processed successfully!"}


