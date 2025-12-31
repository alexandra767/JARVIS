"""
Alexandra RAG System - Document Search & Retrieval
Supports: PDF, TXT, JSON documents
Uses: ChromaDB + Sentence Transformers
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using ChromaDB default embeddings")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyPDF2 not available, PDF support disabled")


class AlexandraRAG:
    """RAG system for Alexandra AI document search"""
    
    def __init__(self,
                 persist_dir: str = None,
                 collection_name: str = "documents",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system

        Args:
            persist_dir: Directory to store ChromaDB data
            collection_name: Name of the document collection
            embedding_model: Sentence transformer model for embeddings
        """
        # Use expanduser for proper home directory resolution
        if persist_dir is None:
            persist_dir = os.path.expanduser("~/ai-clone-chat/rag_db")
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Create persist directory
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        if self.embedder:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Alexandra AI document store"}
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Alexandra AI document store"}
            )
        
        print(f"RAG initialized. Documents in collection: {self.collection.count()}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text = text.strip()
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        return chunks
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not installed. Run: pip install pypdf2")
        
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file for deduplication"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def add_document(self, 
                     filepath: str, 
                     metadata: Optional[Dict] = None,
                     chunk_size: int = 1000) -> int:
        """
        Add a document to the RAG system
        
        Args:
            filepath: Path to document (PDF, TXT, JSON)
            metadata: Optional metadata to attach
            chunk_size: Size of text chunks
            
        Returns:
            Number of chunks added
        """
        filepath = os.path.abspath(filepath)
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Extract text based on file type
        if file_ext == '.pdf':
            text = self._extract_pdf_text(filepath)
        elif file_ext in ['.txt', '.md']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif file_ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
        else:
            # Try to read as text
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        if not text.strip():
            print(f"Warning: No text extracted from {filename}")
            return 0
        
        # Chunk the text
        chunks = self._chunk_text(text, chunk_size=chunk_size)
        
        if not chunks:
            return 0
        
        # Prepare metadata
        file_hash = self._get_file_hash(filepath)
        base_metadata = {
            "source": filename,
            "filepath": filepath,
            "file_hash": file_hash,
            "file_type": file_ext,
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Generate embeddings and add to collection
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_hash}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({**base_metadata, "chunk_index": i})
            
            if self.embedder:
                embedding = self.embedder.encode(chunk).tolist()
                embeddings.append(embedding)
        
        # Add to ChromaDB
        if embeddings:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        
        print(f"Added {len(chunks)} chunks from {filename}")
        return len(chunks)
    
    def add_directory(self, 
                      directory: str, 
                      extensions: List[str] = ['.pdf', '.txt', '.md', '.json'],
                      recursive: bool = True) -> int:
        """
        Add all documents from a directory
        
        Args:
            directory: Path to directory
            extensions: File extensions to include
            recursive: Search subdirectories
            
        Returns:
            Total chunks added
        """
        total_chunks = 0
        directory = Path(directory)
        
        if recursive:
            files = []
            for ext in extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            files = []
            for ext in extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        print(f"Found {len(files)} documents to process")
        
        for i, filepath in enumerate(files):
            try:
                chunks = self.add_document(str(filepath))
                total_chunks += chunks
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(files)} files...")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        print(f"Total: Added {total_chunks} chunks from {len(files)} files")
        return total_chunks
    
    def search(self,
               query: str,
               n_results: int = 5,
               filter_source: Optional[str] = None,
               keyword_search: bool = True) -> List[Dict]:
        """
        Search for relevant documents using hybrid search (semantic + keyword)

        Args:
            query: Search query
            n_results: Number of results to return
            filter_source: Optional filter by source filename
            keyword_search: Also filter by keyword match (better for names)

        Returns:
            List of results with text, metadata, and scores
        """
        # Build metadata filter
        where_filter = None
        if filter_source:
            where_filter = {"source": {"$contains": filter_source}}

        # Build document text filter for keyword matching (case insensitive)
        where_document = None
        if keyword_search and query.strip():
            # For names/keywords, search for the term in document text
            where_document = {"$contains": query.lower()}

        # Try keyword search first for better name matching
        if keyword_search:
            try:
                keyword_results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_filter,
                    where_document=where_document,
                    include=["documents", "metadatas", "distances"]
                )
                # If we got results with keyword match, use those
                if keyword_results and keyword_results.get('documents') and keyword_results['documents'][0]:
                    results = keyword_results
                else:
                    # Fall back to semantic search
                    if self.embedder:
                        query_embedding = self.embedder.encode(query).tolist()
                        results = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=n_results,
                            where=where_filter,
                            include=["documents", "metadatas", "distances"]
                        )
                    else:
                        results = self.collection.query(
                            query_texts=[query],
                            n_results=n_results,
                            where=where_filter,
                            include=["documents", "metadatas", "distances"]
                        )
            except Exception as e:
                print(f"Keyword search failed, using semantic: {e}")
                if self.embedder:
                    query_embedding = self.embedder.encode(query).tolist()
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=where_filter,
                        include=["documents", "metadatas", "distances"]
                    )
                else:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        where=where_filter,
                        include=["documents", "metadatas", "distances"]
                    )
        # Semantic-only search
        elif self.embedder:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        
        # Format results
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    'text': results['documents'][0][i],
                    'source': results['metadatas'][0][i].get('source', 'Unknown'),
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted
    
    def search_with_context(self, query: str, n_results: int = 3) -> str:
        """
        Search and return formatted context for LLM
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            Formatted context string
        """
        results = self.search(query, n_results=n_results)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[Source {i}: {r['source']}]\n{r['text']}\n")
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "total_chunks": self.collection.count(),
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name
        }
    
    def list_sources(self) -> List[str]:
        """List all unique document sources"""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in results['metadatas']:
            if 'source' in meta:
                sources.add(meta['source'])
        return sorted(list(sources))
    
    def delete_source(self, source_name: str) -> int:
        """Delete all chunks from a source"""
        results = self.collection.get(
            where={"source": source_name},
            include=["metadatas"]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0
    
    def clear_all(self):
        """Clear all documents from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Alexandra AI document store"}
        )
        print("Collection cleared")


# Singleton instance
_rag_instance = None

def get_rag() -> AlexandraRAG:
    """Get or create RAG singleton"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AlexandraRAG()
    return _rag_instance


# CLI for testing
if __name__ == "__main__":
    import sys
    
    rag = AlexandraRAG()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "add" and len(sys.argv) > 2:
            path = sys.argv[2]
            if os.path.isdir(path):
                rag.add_directory(path)
            else:
                rag.add_document(path)
        
        elif cmd == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            results = rag.search(query)
            for r in results:
                print(f"\n[{r['source']}] (distance: {r['distance']:.4f})")
                print(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])
        
        elif cmd == "stats":
            print(rag.get_stats())
        
        elif cmd == "sources":
            print("Sources:", rag.list_sources())
        
        elif cmd == "clear":
            rag.clear_all()
    else:
        print("Usage:")
        print("  python alexandra_rag.py add <file_or_directory>")
        print("  python alexandra_rag.py search <query>")
        print("  python alexandra_rag.py stats")
        print("  python alexandra_rag.py sources")
        print("  python alexandra_rag.py clear")
