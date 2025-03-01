import os
from typing import List, Optional
import logging
import sys
import tqdm
from math import ceil
import requests

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.tools import Tool
import chromadb
from chromadb.config import Settings
from langchain.embeddings.base import Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddings(Embeddings):
    """
    Custom embeddings class for local embedding API that's compatible with LangChain
    """
    
    def __init__(
        self,
        base_url: str,
        model: str = "text-embedding-nomic-embed-text-v1.5-embedding",
        request_timeout: Optional[float] = None
    ):
        """Initialize the local embeddings client"""
        self.base_url = base_url
        self.model = model
        self.request_timeout = request_timeout
        self.embedding_url = f"{base_url}/embeddings"
        self.headers = {"Content-Type": "application/json"}
        logger.info(f"Initialized LocalEmbeddings with model {model} at {base_url}")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents/texts"""
        # Ensure all inputs are strings
        texts = [str(text) for text in texts]
        
        # Prepare the API request
        payload = {
            "input": texts,
            "model": self.model
        }
        
        # Make the API request
        try:
            response = requests.post(
                self.embedding_url, 
                headers=self.headers, 
                json=payload,
                timeout=self.request_timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                embeddings = [item.get("embedding", []) for item in result.get("data", [])]
                return embeddings
            else:
                raise ValueError(f"API returned error: {response.text}")
                
        except Exception as e:
            logger.error(f"Error during embedding request: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text query"""
        # Reuse embed_documents for single query
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

class RAGSystem:
    """
    RAG (Retrieval Augmented Generation) system that processes markdown files,
    stores them in a local vector database, and provides retrieval capabilities.
    """
    
    def __init__(
        self, 
        base_url: str,
        vectorstore: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
        max_batch_size: int = 1000  # Maximum batch size for adding documents
    ):
        """
        Initialize the RAG system.
        
        Args:
            vectorstore: Directory to persist the vector database (required)
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            max_batch_size: Maximum batch size for adding documents to the vector store
        """
        self.persist_directory = vectorstore
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_batch_size = max_batch_size
        
        # Initialize embeddings
        logger.info("Initializing embeddings")
        self.embeddings = LocalEmbeddings(
            base_url=base_url,
            model="text-embedding-nomic-embed-text-v1.5-embedding"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize or load vector store
        # Create Chroma client with telemetry disabled
        chroma_client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path=self.persist_directory)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            client=chroma_client
        )
    
    def add_documents_in_batches(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents to add")
            return
            
        # Calculate number of batches
        num_docs = len(documents)
        num_batches = ceil(num_docs / self.max_batch_size)
        logger.info(f"Adding {num_docs} documents in {num_batches} batches (max batch size: {self.max_batch_size})")
        
        # Create progress bar
        with tqdm.tqdm(total=num_docs, desc="Adding documents to vector store") as pbar:
            for i in range(num_batches):
                start_idx = i * self.max_batch_size
                end_idx = min((i + 1) * self.max_batch_size, num_docs)
                batch = documents[start_idx:end_idx]
                
                logger.info(f"Adding batch {i+1}/{num_batches} with {len(batch)} documents")
                self.vector_store.add_documents(batch)
                
                # Update progress bar
                pbar.update(len(batch))
                
        logger.info(f"Successfully added {num_docs} documents to vector store")
    
    def process_markdown_directory(self, directory_path: str) -> int:
        logger.info(f"Processing markdown files in {directory_path}")
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory {directory_path} does not exist")
            return 0
        
        try:
            # First try with UnstructuredMarkdownLoader
            logger.info("Attempting to use UnstructuredMarkdownLoader")
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader,
                show_progress=True
            )
            
            # Load documents
            logger.info(f"Loading documents from {directory_path}")
            try:
                documents = loader.load()
            except ImportError as e:
                logger.warning(f"UnstructuredMarkdownLoader failed: {e}. Falling back to TextLoader")
                loader = DirectoryLoader(
                    directory_path,
                    glob="**/*.md",
                    loader_cls=TextLoader,
                    show_progress=True
                )
                documents = loader.load()
            
            if not documents:
                logger.warning(f"No documents found in {directory_path}")
                return 0
                
            logger.info(f"Loaded {len(documents)} documents")
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            
            # Add documents to vector store in batches
            self.add_documents_in_batches(split_docs)
            
            # Note about persistence
            logger.info(f"Documents are automatically persisted to {self.persist_directory}")
            
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return 0
    
    def query(self, query: str, k: int = 5) -> List[Document]:
        logger.info(f"Querying vector store with: {query}")
        return self.vector_store.similarity_search(query, k=k)
    
    def get_retrieval_tool(self) -> Tool:
        def retrieve_relevant_documents(query: str) -> str:
            """Retrieve relevant documents for a query."""
            docs = self.query(query)
            if not docs:
                return "No relevant information found."
           
            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content.replace("\n", " ").strip()
                results.append(f"[{i}] From {source}:\n{content}\n")
            
            return "\n".join(results)
        
        return Tool(
            name="DocumentSearch",
            func=retrieve_relevant_documents,
            description="Useful for searching information in the loaded documents. Input should be a search query."
        )

def process_documents(args):
    """Process documents and store them in the vector database."""
    try:
        # Get max_batch_size from args if available, otherwise use default
        max_batch_size = getattr(args, 'max_batch_size', 1000)
        
        rag = RAGSystem(
            base_url=args.base_url,
            vectorstore=args.vectorstore,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_batch_size=max_batch_size
        )
        
        num_docs = rag.process_markdown_directory(args.directory)
        print(f"Successfully processed {num_docs} documents")
        print(f"Vector database automatically stored in: {args.vectorstore}")
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        print(f"Failed to process documents: {e}")
        sys.exit(1)

def query_documents(args):
    """Query the vector database directly without the agent."""
    try:
        # Get max_batch_size from args if available, otherwise use default
        max_batch_size = getattr(args, 'max_batch_size', 5400)
        
        rag = RAGSystem(
            base_url=args.base_url,
            vectorstore=args.vectorstore,
            max_batch_size=max_batch_size
        )
        
        results = rag.query(args.query, k=args.k)
        
        if not results:
            print("No relevant documents found.")
            return
            
        print(f"\nFound {len(results)} relevant document chunks:\n")
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"\n[{i}] From: {source}")
            print(f"{doc.page_content}")
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        print(f"Failed to query documents: {e}")
        sys.exit(1)
