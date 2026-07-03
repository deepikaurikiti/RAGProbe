import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

def index_data():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found. Please run fetch_data.py first.")
        return

    print("Loading documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found to index.")
        return
        
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing Ollama embedding model (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving index to '{INDEX_DIR}'...")
    vectorstore.save_local(INDEX_DIR)
    print("Indexing complete.")

if __name__ == "__main__":
    index_data()
