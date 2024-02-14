from fastapi import FastAPI, Request, Form
from typing import Optional
from chromadb.client import ChromaClient  # Assuming you have ChromaDB installed

app = FastAPI()
chroma_client = ChromaClient("localhost", port=9000)  # Replace with your ChromaDB details

@app.post("/upload_pdf")
async def upload_pdf(pdf_path: str = Form(...), pdf_name: str = Form(...)):
    """
    Uploads a PDF, converts it to embeddings, and saves them to ChromaDB.

    Args:
        pdf_path: Path to the PDF file.
        pdf_name: Name of the PDF document.

    Returns:
        Success message or error information.
    """

    try:
        # Open and read the PDF content
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()

        # Extract text from the PDF (you might need an external library for complex PDFs)
        text = extract_text(pdf_content)

        # Segment text into smaller chunks (adjust window size as needed)
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

        # Generate embeddings for each chunk
        embeddings = [generate_embedding(chunk) for chunk in chunks]

        # Save embeddings to ChromaDB
        for i, embedding in enumerate(embeddings):
            chroma_client.put_data(f"{pdf_name}_chunk_{i}", embedding)

        return {"message": "PDF document successfully uploaded and embeddings saved to ChromaDB."}

    except Exception as e:
        return {"error": f"Error uploading PDF: {str(e)}"}

