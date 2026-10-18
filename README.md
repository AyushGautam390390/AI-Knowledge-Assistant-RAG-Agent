# AI Knowledge Assistant — RAG Agent

An AI-powered question-answering application that uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from PDF documents and generate contextual answers using a local LLM.

## Features

- **PDF Document Processing:** Loads and processes PDF files from a local directory.
- **Text Chunking:** Splits documents into smaller chunks for retrieval.
- **Semantic Search:** Uses Sentence Transformers and FAISS for vector similarity search.
- **LLM Integration:** Uses Ollama with the `llama3.2:1b` model.
- **Agentic Workflow:** Uses LangChain tools and an agent to retrieve relevant document context.
- **REST API:** Exposes a question-answering endpoint through FastAPI.
- **Docker Support:** Packages the application and its dependencies into a Docker image.

## Tech Stack

- Python 3.11
- LangChain
- FAISS
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Ollama (`llama3.2:1b`)
- FastAPI
- Docker

## Project Structure

```text
rag_agent/
├── books/              # Source PDF documents
├── app.py              # FastAPI application
├── rag_agent.py        # RAG pipeline and agent
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image configuration
├── .dockerignore       # Files excluded from Docker builds
└── README.md           # Project documentation
```

## How It Works

1. Loads PDF documents from the `books/` directory.
2. Splits the documents into overlapping text chunks.
3. Converts chunks into vector embeddings using Sentence Transformers.
4. Stores embeddings in a FAISS vector index.
5. Retrieves relevant chunks when a user asks a question.
6. Uses the LangChain agent and Ollama LLM to generate a response.
7. Returns the answer through the FastAPI endpoint.

## Run Locally

### Prerequisites

- Python 3.11
- Ollama installed and running
- Git installed

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd rag_agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the LLM

```bash
ollama pull llama3.2:1b
```

Ensure `rag_agent.py` uses `http://localhost:11434` for local execution.

### 5. Start the API

```bash
python -m uvicorn app:app --reload
```

Open `http://localhost:8000/docs` to test the API.

## Run with Docker

Ensure Docker Desktop and Ollama are running on the host machine.

Build the image:

```bash
docker build -t rag-agent .
```

Run the container:

```bash
docker run --name rag-agent-container -p 8000:8000 rag-agent
```

The Docker configuration should use `http://host.docker.internal:11434` to connect to Ollama running on the Windows host.

Open `http://localhost:8000/docs` to interact with the API.

## API Usage

**Endpoint:** `POST /ask`

Example request:

```json
{
  "question": "What is an AI agent?"
}
```

The API returns the submitted question and the generated answer.

## Notes

- The first startup may take time while the embedding model is downloaded and initialized.
- Ollama runs separately from the Docker container unless you configure it as a separate service.
- Responses depend on the retrieved documents and the underlying language model.
- Do not upload confidential PDFs, API keys, access tokens, or environment files to a public repository.

## Future Improvements

- Add persistent FAISS index storage to reduce startup time.
- Add source citations to answers.
- Add conversation history and memory.
- Add a Streamlit frontend.
- Deploy the API to a cloud platform.

## License

Choose an appropriate open-source license if you intend to allow others to reuse the project.
