from typing import Annotated,Optional
import chromadb
import os
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pdfrw import PdfReader
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, Form, UploadFile,HTTPException
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from chromadb import Documents, EmbeddingFunction, Embeddings
#.ENV is the file for API key from makersuite
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
app = FastAPI()
retriever=None
chroma_client = chromadb.Client()
#it is a globally declared string to store the text from the pdf after processing, i have used this to pass text in a function
global_text=None
#----------------------------------------------------------------------------------#

@app.post("/store-pdf-in-chroma/")
async def store_pdf_in_chroma(pdf_path, pdf_name: Optional[str] = None):
    """#API will take pdf_path and pdf name as input and should
      return a success message if the document is successfully 
      converted to embedding and saved to Chroma Db."""
    # pdf_path="C:/Users/promact.DESKTOP-RHBFB7T/Downloads/docu.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found.")
    
    try:
        # Read PDF and extract text
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        return {"message": f"PDF document '{pdf_name}' stored in ChromaDB with document ID ."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing PDF document in ChromaDB: {str(e)}")
    #---------------------------------------------------------------------------------------------------#
#This is the another way to upload pdf, i have performed the main algorithm using this method
@app.post("/upload-pdfs/")
async def create_file(pdf_name=str,
    files: list[UploadFile] = Form(...)
):
    """
    Uploads a PDF, converts it to embeddings, and saves them to ChromaDB.

    Args:
        pdf_name: Pdf Name
        pdf_fiel: upload file

    Returns:
        Success message or error information.
    """
    #local pdf directory will be created named Docs
    pdf_store = "Docs"
    os.makedirs(pdf_store, exist_ok=True)
    print("You have successfully created pdf directory")
    #Three empty lists are created to store some necessary data
    uploaded_documents = []
    extracted_texts = []
    embeddings = []
    #traversing file from multiple files
    for file in files:
        #testing wheter the uploaded document is pdf or not
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            return {"error": "Invalid file format. Please upload only PDFs."}
        file_name = file.filename
        # file_path=f"{pdf_store}/file_name" 
        
        #storing name in a varible
        # file_name = file.filename
        # Construct the file path to save the uploaded file
        file_path = os.path.join(pdf_store, file_name)
        # file_path=f"{pdf_store}/file_name"
        try:
            # Read PDF using PDF Reader
            documents = [file.read().decode()]
            pdf_reader = PdfReader(file.file)
            #Empty text string
            text = ""
            #For multiple pages in the pdf this loop travers to all pages and add the extracted text to "text"
            for page in pdf_reader.pages:
                text += page.ExtractText()
        except Exception as e:
            return {"error": f"Error processing file {file.filename}: {e}"}
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        #passing the combined text in text_catcher to store in a global variable
        text_catcher(texts)
    
        #Task 2: Embedding and Chroma Vector Database
        #Function call for embedding
        # database=embeddingfun(texts)
        #Using Google PalmEmbedding to convert text into embedded form and store it in chroma db
        embeddings=GooglePalmEmbeddings(model="models/embedding-001")
         # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # db = chroma_client.create_collection(name=name, embeddings))
        retriever = db.as_retriever()
    return  f"Document '{pdf_name}' has been successfully imported to the Chroma database."

#Method to ask question 
@app.post("/ask-question/")
async def user_question(pdf_name: str, question: str):
    try:
      #global text is the collective text of the pdf passing here to search from chroma db
      text_doc=global_text
      similar_segment = vector_search(question,text_doc)
      answer = get_conversational_chain(similar_segment, question)
      return {"answer": answer}
    #Exception message will be displayed
    except Exception as e:
      return {"error": f"Error in anwering the questions {e}"}
    
#This function wil give relevant or most similar result
def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage
#Similarity Search will be done based on the user question and calculate the result.
def vector_search(question,text_doc):
    data=embeddingfun(text_doc)
    result = retriever.get_relevant_passage(question,data)
    return result

#This is the conversational_chain
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


      model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
      prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
      chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      response = chain({"input_documents":result, "question": question}, return_only_outputs=True)
      return response
    except Exception as e:
       return f"Error arrived: {str(e)}"
    
#There is basically two times embeddings happening because i have not declared embedding globally
def embeddingfun(texts):
        embeddings=GooglePalmEmbeddings(model="models/embedding-001")
         # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        return db

#This function is made to store the collective text to global_text to pass in a function
def text_catcher(texts):
    global_text=texts
    return texts