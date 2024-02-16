from typing import Annotated
import chromadb
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pdfrw import PdfReader
from PyPDF2 import PdfReader, PyPDFLoader
from fastapi import FastAPI, File, Form, UploadFile
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from chromadb import Documents, EmbeddingFunction, Embeddings

app = FastAPI()
retriever=None
chroma_client = chromadb.Client()
global_text=None
#.ENV is the file for API keys 
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#Method to upload files
@app.post("/upload-pdfs/")
async def create_file(pdf_name=str,
    files: list[UploadFile] = Form(...)
):
    pdf_store = "docs"
    os.makedirs(pdf_store, exist_ok=True)
    print("You have successfully created pdf directory")
    uploaded_documents = []
    extracted_texts = []
    embeddings = []
    for file in files:
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            return {"error": "Invalid file format. Please upload only PDFs."}
        file_name = file.filename
        file_path=f"{pdf_store}/file_name"
        try:
            # Read PDF using PDF PD2
            documents = [file.read().decode()]
            pdf_reader = PdfReader(file.file)
            text = ""
            for page in pdf_reader.pages:
                text += page.ExtractText()
        except Exception as e:
            return {"error": f"Error processing file {file.filename}: {e}"}
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        text_catcher(texts)
        
        #2nd Way

        loa_doc = PyPDFLoader(file.filename)
        context = loa_doc.load()
        chunks = text_splitter.split_documents(context)
        database=embeddingfun(texts)
        embeddings=GooglePalmEmbeddings(model="models/embedding-001")
         # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # db = chroma_client.create_collection(name=name, embeddings))
        retriever = db.as_retriever()
    return  f"Document '{pdf_name}' File is succesfully imported to chroma db"

@app.post("/ask-question/")
async def user_question(pdf_name: str, question: str):
    try:
      text_doc=global_text
      similar_segment = vector_search(question,text_doc)
      answer = get_conversational_chain(similar_segment, question)
      return {"answer": answer}
    except Exception as e:
      return {"error": f"Error answering question or answer not found: {e}"}
def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage

def vector_search(question,text_doc):
    data=embeddingfun(text_doc)
    result = retriever.get_relevant_passage(question,data)
    return result
# def create_chroma_db(documents, name):
#   chroma_client = chromadb.Client()
#   db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

#   for i, d in enumerate(documents):
#     db.add(
#       documents=d,
#       ids=str(i)
#     )
#   return db
# def get_text_chunks(text):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks=text_splitter.split_text(text)
#     return chunks
def get_conversational_chain(result,question):
    prompt_template="""Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the 
    porovided context just say,"anser is not available in the context", don't provide the wrong answer \n\n
    Context:\n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    try:
      if not result:
        return "Answer not available in the context."


      model = ChatGoogleGenerativeAI(model="gemini-ultra", temperature=0.2)
      prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
      chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      response = chain({"input_documents":result, "question": question}, return_only_outputs=True)
      return response
    except Exception as e:
       return f"Error arrived: {str(e)}"
    
def embeddingfun(texts):
        embeddings=GooglePalmEmbeddings(model="models/embedding-001")
         # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        return db
def text_catcher(texts):
    global_text=texts
    return texts