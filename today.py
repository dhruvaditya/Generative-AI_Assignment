from fastapi import FastAPI, Form, HTTPException
from chroma_db import ChromaDB  # Assuming Chroma DB library is implemented separately
from pdf_processing import extract_text_from_pdf, convert_text_to_embeddings

app = FastAPI()
chroma_db = ChromaDB()  # Initialize Chroma DB instance

def segment_text(text, segment_length=1000):
    # Segment text into chunks of specified length
    segments = []
    for i in range(0, len(text), segment_length):
        segments.append(text[i:i+segment_length])
    return segments

@app.post('/convert_pdf_to_embedding')
async def convert_pdf_to_embedding(pdf_path: str = Form(...), pdf_name: str = Form(...)):
    # Extract text from PDF
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Segment text into smaller chunks
    segments = segment_text(text)

    # Convert each segment to embeddings and save to Chroma DB
    for i, segment in enumerate(segments):
        try:
            embeddings = convert_text_to_embeddings(segment)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Save embeddings to Chroma DB
        try:
            chroma_db.save_embeddings(f"{pdf_name}_segment_{i+1}", embeddings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {'message': 'Document successfully converted and saved to Chroma DB.'}
