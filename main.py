from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import process_pdf, ask_question
import shutil
import os

app = FastAPI()

# Add CORS middleware to allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development (change to specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    message = process_pdf(file_path)
    return {"message": message, "filename": file.filename}


@app.get("/ask")
async def query_pdf(question: str):
    answer = ask_question(question)
    return {"question": question, "answer": answer}