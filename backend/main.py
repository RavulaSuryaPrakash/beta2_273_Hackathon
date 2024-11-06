
# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict
import os
from tqdm import tqdm
import time

# Updated imports - removed AutoModelForSeq2SeqGeneration
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

# Get the current directory
CURRENT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = CURRENT_DIR / "documents"
CHROMA_DB_DIR = CURRENT_DIR / "chroma_db"

app = FastAPI()



# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
qa_chain = None
db = None

# Custom prompt template
CUSTOM_PROMPT = PromptTemplate(
    template="""You are an expert analyst of the Global Report on Food Crises. Use the following pieces of context to answer the question. If you don't know the answer, just say "I cannot find specific information about this in the document."

Context: {context}

Question: {question}

Give a detailed answer based on the context provided. Include specific data, statistics, and findings if available. If the information isn't in the context, say so clearly:""",
    input_variables=["context", "question"]
)

def initialize_qa_system():
    global qa_chain, db
    
    try:
        start_time = time.time()
        print("Starting QA system initialization...")
        
        # First, clear any existing ChromaDB data
        if os.path.exists(str(CHROMA_DB_DIR)):
            import shutil
            shutil.rmtree(str(CHROMA_DB_DIR))
        
        # Check if PDF exists
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No PDF files found in documents directory")
        
        print(f"Loading PDF: {pdf_files[0]}")
        
        # Load and split the document
        loader = PyPDFLoader(str(pdf_files[0]))
        documents = loader.load()[:10]
        print(f"Loaded {len(documents)} pages")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        
        # Initialize language model with simplified configuration
        print("Initializing language model...")
        model_pipeline = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
            clean_up_tokenization_spaces=True
        )

        llm = HuggingFacePipeline(
            pipeline=model_pipeline
        )
        
        # Initialize QA chain
        print("Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={
                "prompt": CUSTOM_PROMPT,
                "verbose": True
            },
            return_source_documents=False
        )
        
        end_time = time.time()
        print(f"QA system initialized successfully in {end_time - start_time:.2f} seconds!")
        return True
        
    except Exception as e:
        print(f"Error initializing QA system: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    success = initialize_qa_system()
    if not success:
        print("Failed to initialize QA system")

@app.post("/query")
async def query_document(query: Dict[str, str]):
    global qa_chain, db
    
    if not qa_chain:
        raise HTTPException(
            status_code=503, 
            detail="QA system not initialized"
        )
    
    try:
        question = query.get("text", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Query text is required")


        # Get relevant documents
        docs = db.similarity_search(question, k=4)
        context = "\n".join([doc.page_content for doc in docs])
         # Create input for QA chain
        chain_input = {
            "query": question,
            "context": context
        }
        # Get response from QA chain
        response = qa_chain.invoke(chain_input)
        
        # Extract the answer
        if isinstance(response, dict):
            answer = response.get("result", "")
        else:
            answer = str(response)

        # Ensure meaningful response
        if not answer or len(answer.strip()) < 100:
            answer = "I cannot find specific information about this in the document."

        return {"response": answer}
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question. Please try again."
        )



@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "qa_system": "initialized" if qa_chain else "not initialized",
        "document_loaded": bool(list(DOCUMENTS_DIR.glob("*.pdf")))
    }

@app.get("/status")
async def get_status():
    return {
        "status": "ready" if qa_chain else "initializing",
        "system_ready": qa_chain is not None,
        "document_loaded": bool(list(DOCUMENTS_DIR.glob("*.pdf")))
    }
@app.post("/rebuild-database")
async def rebuild_database():
    """Force rebuild the vector database"""
    try:
        print("Removing existing ChromaDB...")
        if os.path.exists(str(CHROMA_DB_DIR)):
            import shutil
            shutil.rmtree(str(CHROMA_DB_DIR))
            print("Existing ChromaDB removed")
        
        print("Reinitializing QA system...")
        success = initialize_qa_system()
        return {
            "status": "success" if success else "failed",
            "message": "Database rebuilt successfully" if success else "Failed to rebuild database"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error rebuilding database: {str(e)}"
        )