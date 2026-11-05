import os
import uuid

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentParserService:
    """
    Handles extracting text from raw files (PDFs, text files),
    cleaning text, chunking it with semantic awareness, and attaching RBAC metadata.
    """
    def __init__(self, chunk_size: int= 800, chunk_overlap: int = 150):

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Extracts raw pages from a PDF file using PyPDFLoader.
        Each page becomes an initial LangChain Document with metadata: {'source': ..., 'page': ...}
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")

        loader = PyPDFLoader(file_path=file_path)

        pages = loader.load()
        return pages

    
    def chunk_documents(
        self,
        documents: List[Document],
        document_id: Optional[str] = None,
        role_access: Optional[List[str]] = None
    ) -> List[Document]:

        doc_id = document_id or str(uuid.uuid4())

        roles = role_access or ["user", "admin"]

        raw_chunks = self.text_splitter.split_documents(documents)

        enriched_chunks: List[Document] = []

        for index, chunk in enumerate(raw_chunks):
            cleaned_content = chunk.page_content.replace("\x00", "").strip()
            if not cleaned_content:
                continue
            metadata: Dict[str, Any] = dict(chunk.metadata)
            metadata["document_id"] = doc_id
            metadata["chunk_index"] = index
            metadata["chunk_id"] = f"{doc_id}_{index}"
            metadata["role_access"] = roles
            metadata["page"] = metadata.get("page", 0) + 1
            enriched_chunk = Document(
                page_content=cleaned_content,
                metadata=metadata
            )
            enriched_chunks.append(enriched_chunk)
        print(f"[DocumentParserService] Successfully created {len(enriched_chunks)} chunks from document ID: {doc_id}")
        return enriched_chunks 