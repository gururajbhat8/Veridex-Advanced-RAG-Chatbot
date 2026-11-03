
# Veridex - Enterprise Multimodal RAG Chatbot

An enterprise-ready Retrieval-Augmented Generation (RAG) platform supporting multimodal documents (text, diagrams, and images), role-based access control (RBAC), multi-turn conversation memory, and automated RAGAS quality evaluation.

---

## Architecture Overview

```
[ User / Browser ]
        │
        ▼
[ Streamlit UI ] ──(HTTP/REST)──▶ [ FastAPI Backend ]
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
[ AWS Cognito ]                   [ LangChain Core ]               [ S3 Document Store ]
(Auth & RBAC)                            │                         (Source of Truth)
                                         ├──────────────────┐
                                         ▼                  ▼
                              [ Pinecone Vector DB ]   [ DynamoDB ]
                              (Metadata Role Filtering) (Chat History)
                                         │
                                         ▼
                               [ LLM / Vision Engine ]
                               - AWS Bedrock (Claude / Nova / Llama)
                               - Fallback: Groq / Local LLM
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Conversational UI, document upload, and multimodal preview |
| **Backend API** | FastAPI | Asynchronous RESTful API services |
| **RAG Orchestration** | LangChain / LangGraph | Chunking, prompt templates, retrieval chains, and agentic workflows |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Free, local, high-performance dense embeddings |
| **Vector Database** | Pinecone | Managed vector search with metadata filtering for RBAC |
| **Document Store** | AWS S3 | Encrypted source of truth for raw PDF/document files |
| **Chat Memory** | AWS DynamoDB | Persistent, low-latency multi-turn conversation history |
| **LLMs & Vision** | AWS Bedrock / Groq | Text generation and multimodal image/diagram captioning |
| **Authentication & RBAC** | AWS Cognito | Role-based identity management (Admin, Manager, User) |
| **Evaluation & Tracing** | RAGAS & Langfuse | Faithfulness, context recall/precision, and latency tracing |
| **Containerization** | Docker & Docker Compose | Multi-container setup for local development and deployment |

---

## Project Structure

```
Veridex-Advanced-RAG-Chatbot/
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── Dockerfile                   # Docker image configuration
├── docker-compose.yml           # Multi-service container orchestration
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
│
├── frontend/                    # Streamlit Frontend application
│   ├── app.py                   # Streamlit main entry point
│   └── components/              # Reusable UI widgets
│       ├── chat_interface.py    # Chat window & stream handler
│       ├── doc_viewer.py        # Document & image viewer
│       └── sidebar.py           # Configuration & file uploader
│
├── src/                         # FastAPI Backend application
│   ├── main.py                  # FastAPI application entry point
│   ├── api/                     # API routing layer
│   │   └── routes/
│   │       ├── auth.py          # Cognito authentication endpoints
│   │       ├── chat.py          # RAG query & streaming endpoints
│   │       ├── documents.py     # Document upload & ingestion triggers
│   │       └── health.py        # Health check endpoint
│   │
│   ├── core/                    # Core application configurations
│   │   ├── config.py            # Pydantic BaseSettings & environment configs
│   │   └── security.py          # Token validation & RBAC dependencies
│   │
│   ├── models/                  # Pydantic schemas (Data Transfer Objects)
│   │   ├── chat.py              # Query & response models
│   │   └── document.py          # Document metadata & ingestion schemas
│   │
│   └── services/                # Business logic & external service integrations
│       ├── evaluation/
│       │   └── evaluator.py     # RAGAS metrics & evaluation pipeline
│       ├── ingestion/
│       │   ├── embedder.py      # HuggingFace embeddings wrapper
│       │   ├── parser.py        # Multimodal parser (PDF text + images)
│       │   └── vector_store.py  # Pinecone indexing & upsert logic
│       ├── rag/
│       │   ├── chain.py         # Conversational RAG chain
│       │   ├── llm.py           # Bedrock / Groq client factories
│       │   └── retriever.py     # Role-aware Pinecone retriever
│       └── storage/
│           ├── dynamodb.py      # DynamoDB chat session store
│           └── s3.py            # S3 client for document & image storage
│
└── tests/                       # Automated test suite
    ├── test_api.py              # API endpoint integration tests
    ├── test_ingestion.py        # Parsing & chunking unit tests
    └── test_rag.py              # Retrieval & response generation tests
```

---

## Phased Development Roadmap

1. **Phase 1: Environment & Project Architecture Setup**
2. **Phase 2: Local Ingestion Pipeline & Vector Store (HuggingFace + Pinecone)**
3. **Phase 3: AWS S3 Document Storage (Source of Truth)**
4. **Phase 4: Core RAG Engine & Free Fallback LLM (FastAPI + Groq/Ollama)**
5. **Phase 5: DynamoDB Chat History & Session Management**
6. **Phase 6: AWS Bedrock Integration (Claude / Nova / Llama)**
7. **Phase 7: Multimodal Parsing (Images, Charts & Diagrams)**
8. **Phase 8: Streamlit Frontend & Enterprise UI Features**
9. **Phase 9: Role-Based Access Control (RBAC) & Cognito Auth**
10. **Phase 10: RAGAS Quality Evaluation & Observability**
11. **Phase 11: Containerization & Deployment**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

