from fastapi import FastAPI, UploadFile, File,Form
from secrets import token_hex
import uvicorn

app = FastAPI(title="Upload the pdf file")

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    file_ext = file.filename.split(".").pop()
    file_name = token_hex(10)
    file_path = f"{file_name}.{file_ext}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"success": True, "file_path": file_path, "message": "File uploaded successfully"}
@app.post("/process_pdf/")
async def process_pdf(pdf_name: str = Form(...), question: str = Form(...)):
    # Do something with the provided PDF name and question
    return {"pdf_name": pdf_name, "question": question}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
