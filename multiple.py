from fastapi import FastAPI, File, UploadFile, Form
from pdfrw import PdfReader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch
import chromadb

app = FastAPI()

# Replace with your specific API key and access details
CHROMA_ACCESS_KEY = "YOUR_KEY"
ELASTICSEARCH_HOST = "localhost"  # Or your Elasticsearch host URL
ELASTICSEARCH_INDEX = "my_pdf_index"

# Load pretrained language model and sentence transformer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
sentence_transformer = SentenceTransformer(model_name)

# Create Elasticsearch client and ChromaDB instance
es_client = Elasticsearch(host=ELASTICSEARCH_HOST)
chroma_client = chromadb.ChromaDb(CHROMA_ACCESS_KEY)

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

# Similarity search function
@app.post("/search")
async def search(query: str = Form(...)):
    query_embedding = sentence_transformer.encode(query)
    results = chroma_client.search(query_embedding, "vector_collection")

    relevant_texts = []
    for hit in results:
        text = extracted_texts[hit["id"]]
        relevant_texts.append(text)

    return {"query": query, "results": relevant_texts}