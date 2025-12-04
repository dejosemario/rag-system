# RAG System with FastAPI

A complete Retrieval-Augmented Generation (RAG) system built with Python FastAPI, following the provided starter template structure.

## Overview

This project implements a full-featured RAG system with semantic chunking, vector embeddings, and LLM-powered question answering. The implementation uses the official Python FastAPI starter template with three core endpoints: `/upload`, `/prompt`, and `/rechunk`.

## Features

- **Document Upload**: Upload PDF, TXT, or DOCX files via `/upload` endpoint
- **Semantic Search**: Query documents using natural language via `/prompt` endpoint
- **Dynamic Re-chunking**: Adjust chunk sizes on-the-fly via `/rechunk` endpoint
- **Automatic Indexing**: Documents are automatically processed on server startup
- **Vector Embeddings**: Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector Storage**: ChromaDB for efficient similarity search
- **LLM Generation**: Google Gemini Flash 2.5 for intelligent answers

## Technologies Used

- **FastAPI**: Modern Python web framework
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: HuggingFace model for text embeddings
- **Google Gemini**: LLM for answer generation
- **PyPDF2**: PDF document processing
- **python-docx**: DOCX document processing

## Project Structure
```
rag-system/
├── main.py              # Main application (follows starter template)
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Your environment variables (not in git)
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── data/               # Document storage directory
    └── sample.txt      # Sample document for testing
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
HF_API_KEY=your_huggingface_api_key
GEMINI_API_KEY=your_gemini_api_key
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL_NAME=gemini-2.0-flash-exp
RAG_DATA_DIR=./data
CHUNK_LENGTH=500
PORT=8000
```

**Get API Keys:**
- HuggingFace: https://huggingface.co/settings/tokens
- Gemini: https://aistudio.google.com/app/apikey

### 5. Add Documents

Place your documents (PDF, TXT, DOCX) in the `data/` folder:
```bash
mkdir -p data
# Add your documents to data/
```

### 6. Run the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or:
```bash
python main.py
```

The server will:
1. Load all documents from `data/` folder
2. Chunk documents semantically
3. Generate embeddings
4. Store in ChromaDB
5. Start the API server

## API Endpoints

### 1. Upload Document

Upload a new document to the system:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "message": "File 'document.pdf' uploaded and indexed successfully",
  "filename": "document.pdf",
  "size": 12345
}
```

### 2. Query (Prompt)

Ask questions about your documents:
```bash
curl -X POST "http://localhost:8000/prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the vacation policy?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "prompt": "What is the vacation policy?",
  "answer": "Employees receive 15 days of paid vacation per year...",
  "sources": ["company_handbook.pdf"]
}
```

### 3. Re-chunk Documents

Change chunk size and re-process all documents:
```bash
curl -X POST "http://localhost:8000/rechunk" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_length": 300
  }'
```

**Response:**
```json
{
  "message": "Documents re-chunked successfully",
  "old_chunk_length": 500,
  "new_chunk_length": 300,
  "total_chunks": 42
}
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HF_API_KEY` | HuggingFace API key | - | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | - | Yes |
| `EMBED_MODEL_NAME` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | No |
| `LLM_MODEL_NAME` | LLM model | `gemini-2.0-flash-exp` | No |
| `CHROMA_DB_HOST` | ChromaDB host | `localhost:8000` | No |
| `RAG_DATA_DIR` | Data directory | `./data` | No |
| `CHUNK_LENGTH` | Chunk size | `500` | No |
| `PORT` | Server port | `8000` | No |

## How It Works

### 1. Document Ingestion
- Documents are loaded from the `RAG_DATA_DIR` folder
- Supports PDF, TXT, and DOCX formats
- Automatic encoding detection for text files

### 2. Semantic Chunking
- Text is split intelligently by paragraphs first
- Long paragraphs are split by sentences
- Very long sentences are split by character count
- Preserves semantic meaning better than fixed-size chunking

### 3. Embedding Generation
- Each chunk is converted to a 384-dimensional vector
- Uses HuggingFace's sentence-transformers model
- Captures semantic meaning of text

### 4. Vector Storage
- Embeddings stored in ChromaDB
- Efficient similarity search using cosine similarity
- Metadata preserved for source attribution

### 5. Query Processing
- User query converted to embedding
- ChromaDB finds most similar chunks
- Relevant chunks sent to Gemini as context
- LLM generates answer based on context

## Implementation Details

This implementation strictly follows the provided Python FastAPI starter template structure with three core endpoints:

1. **`/upload`**: Handles document uploads with automatic indexing
2. **`/prompt`**: Processes queries and returns AI-generated answers
3. **`/rechunk`**: Allows dynamic adjustment of chunk sizes

### Key Design Decisions

- **Semantic Chunking**: Hierarchical splitting (paragraphs → sentences → characters) preserves context
- **Automatic Startup Indexing**: Documents are indexed when server starts for immediate readiness
- **Error Handling**: Comprehensive exception handling for robustness
- **Flexible Configuration**: All settings configurable via environment variables
- **Source Attribution**: Answers include source document references

## Troubleshooting

### Import Errors
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### API Key Issues
Verify your `.env` file has valid API keys:
```bash
cat .env | grep API_KEY
```

### Port Already in Use
Change the port in `.env`:
```env
PORT=8001
```

### No Documents Found
Ensure documents are in the correct directory:
```bash
ls -la data/
```

## Testing

Test all endpoints:
```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" -F "file=@data/sample.txt"

# Query the system
curl -X POST "http://localhost:8000/prompt" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize the document"}'

# Re-chunk with new size
curl -X POST "http://localhost:8000/rechunk" \
  -H "Content-Type: application/json" \
  -d '{"chunk_length": 300}'
```

## License

MIT License

## Author

Joshua Mario

## Acknowledgments

- Built using the official Python FastAPI RAG starter template
- HuggingFace for sentence-transformers
- Google for Gemini LLM
- ChromaDB for vector storage
- FastAPI for the web framework