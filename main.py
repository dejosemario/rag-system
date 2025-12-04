# main.py - Following the starter template structure
# Endpoints: /upload, /prompt, /rechunk

import os
from pathlib import Path
from typing import List

import chromadb
import docx
import google.generativeai as genai
import PyPDF2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration from environment
HF_API_KEY = os.getenv("HF_API_KEY")
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-exp")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "localhost:8000")
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "./data")
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "500"))
PORT = int(os.getenv("PORT", "8000"))

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with semantic search",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
print(f"Embedding model loaded: {EMBED_MODEL_NAME}")

# Initialize Gemini
print("Configuring Gemini...")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(LLM_MODEL_NAME)
print(f"Gemini model configured: {LLM_MODEL_NAME}")

# Initialize ChromaDB
print("Initializing ChromaDB...")
chroma_client = chromadb.Client()
collection_name = "rag_documents"

collection = chroma_client.get_or_create_collection(
    name=collection_name, metadata={"description": "RAG document embeddings"}
)
print(f"Using collection: {collection_name}")


def load_documents(data_dir: str) -> List[dict]:
    """Load documents from the data directory"""
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Warning: Data directory '{data_dir}' does not exist")
        return documents

    # Load PDF files
    for pdf_file in data_path.glob("**/*.pdf"):
        try:
            print(f"Loading PDF: {pdf_file.name}")
            with open(pdf_file, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                documents.append({"content": text, "source": str(pdf_file.name)})
        except Exception as e:
            print(f"Error loading PDF {pdf_file.name}: {str(e)}")

    # Load TXT files
    for txt_file in data_path.glob("**/*.txt"):
        try:
            print(f"Loading TXT: {txt_file.name}")
            with open(txt_file, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append({"content": text, "source": str(txt_file.name)})
        except UnicodeDecodeError:
            try:
                with open(txt_file, "r", encoding="latin-1") as file:
                    text = file.read()
                    documents.append({"content": text, "source": str(txt_file.name)})
            except Exception as e:
                print(f"Error loading TXT {txt_file.name}: {str(e)}")
        except Exception as e:
            print(f"Error loading TXT {txt_file.name}: {str(e)}")

    # Load DOCX files
    for docx_file in data_path.glob("**/*.docx"):
        try:
            print(f"Loading DOCX: {docx_file.name}")
            doc = docx.Document(str(docx_file))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            documents.append({"content": text, "source": str(docx_file.name)})
        except Exception as e:
            print(f"Error loading DOCX {docx_file.name}: {str(e)}")

    print(f"Loaded {len(documents)} documents")
    return documents


def semantic_chunk_text(text: str, chunk_length: int = 500) -> List[str]:
    """Split text into semantic chunks"""
    chunks = []
    paragraphs = text.split("\n\n")

    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) < chunk_length:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            if len(paragraph) > chunk_length:
                sentences = paragraph.replace("!", ".").replace("?", ".").split(".")
                temp_chunk = ""

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(temp_chunk) + len(sentence) < chunk_length:
                        temp_chunk += sentence + ". "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())

                        if len(sentence) > chunk_length:
                            for i in range(0, len(sentence), chunk_length):
                                chunks.append(sentence[i : i + chunk_length])
                            temp_chunk = ""
                        else:
                            temp_chunk = sentence + ". "

                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                current_chunk = ""
            else:
                current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def index_documents():
    """Load documents, chunk them, create embeddings, and store in ChromaDB"""
    print("Starting document indexing...")

    documents = load_documents(RAG_DATA_DIR)

    if not documents:
        print("No documents found to index")
        return

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc_idx, doc in enumerate(documents):
        chunks = semantic_chunk_text(doc["content"], CHUNK_LENGTH)

        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "source": doc["source"],
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                }
            )
            all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")

    if not all_chunks:
        print("No valid chunks created")
        return

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

    print("Generating embeddings...")
    try:
        embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return

    print("Storing in ChromaDB...")
    try:
        collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas,
            ids=all_ids,
        )
        print(f"Successfully indexed {len(all_chunks)} chunks")
    except Exception as e:
        print(f"Error storing in ChromaDB: {str(e)}")


def query_rag(query: str, top_k: int = 3) -> dict:
    """Query the RAG system"""
    try:
        query_embedding = embedding_model.encode([query])[0]
    except Exception as e:
        print(f"Error encoding query: {str(e)}")
        return {"answer": "Error processing query.", "sources": []}

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {str(e)}")
        return {"answer": "Error searching knowledge base.", "sources": []}

    retrieved_chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not retrieved_chunks:
        return {
            "answer": "No relevant information found.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant. Answer the question based on the following context.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        answer = "Error generating response."

    sources = list(set([meta.get("source", "Unknown") for meta in metadatas]))

    return {"answer": answer, "sources": sources}


# Startup event to index documents on server start
@app.on_event("startup")
async def startup_event():
    """Index documents when server starts"""
    index_documents()


# TEMPLATE ENDPOINTS


@app.post("/upload")
async def upload(file: UploadFile):
    """
    Upload a document to the RAG system.
    The document will be automatically chunked and indexed.
    """
    try:
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Create data directory if it doesn't exist
        data_path = Path(RAG_DATA_DIR)
        data_path.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = data_path / file.filename
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        # Re-index all documents
        index_documents()

        return {
            "message": f"File '{file.filename}' uploaded and indexed successfully",
            "filename": file.filename,
            "size": len(content),
        }
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/prompt")
async def prompt(payload: dict):
    """
    Query the RAG system with a prompt.

    Expected payload:
    {
        "prompt": "Your question here",
        "top_k": 3  (optional, default: 3)
    }
    """
    if "prompt" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'prompt' field")

    query_text = payload["prompt"]
    top_k = payload.get("top_k", 3)

    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        result = query_rag(query_text, top_k)
        return {
            "prompt": query_text,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        print(f"Error processing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/rechunk")
async def rechunk(payload: dict):
    """
    Re-chunk all documents with a new chunk length.

    Expected payload:
    {
        "chunk_length": 500
    }
    """
    if "chunk_length" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'chunk_length' field")

    try:
        new_chunk_length = int(payload["chunk_length"])
        if new_chunk_length < 50 or new_chunk_length > 2000:
            raise HTTPException(
                status_code=400, detail="Chunk length must be between 50 and 2000"
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="Chunk length must be an integer")

    try:
        global CHUNK_LENGTH
        old_chunk_length = CHUNK_LENGTH
        CHUNK_LENGTH = new_chunk_length

        # Delete and recreate collection
        chroma_client.delete_collection(name=collection_name)

        global collection
        collection = chroma_client.create_collection(
            name=collection_name, metadata={"description": "RAG document embeddings"}
        )

        # Re-index with new chunk length
        index_documents()

        doc_count = collection.count()

        return {
            "message": "Documents re-chunked successfully",
            "old_chunk_length": old_chunk_length,
            "new_chunk_length": new_chunk_length,
            "total_chunks": doc_count,
        }
    except Exception as e:
        print(f"Error re-chunking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Re-chunking failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print(f"\nStarting RAG API server on port {PORT}...")
    print(f"API docs available at http://localhost:{PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
