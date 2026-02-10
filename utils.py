import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

vector_db = None

def process_pdf(file_path: str):
    global vector_db
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Store in FAISS (Vector Store)
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(docs, embeddings)
    return "PDF processed successfully!"


def ask_question(query: str):
    if vector_db is None:
        return "Please upload a PDF first."
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )
    response = qa_chain.invoke({"query": query})
    return response["result"]