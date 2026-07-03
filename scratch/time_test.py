import time
import os
import sys
sys.path.append(".")

print("Timing starts...")

t0 = time.time()
import database
print(f"Imported database: {time.time() - t0:.2f}s")

t0 = time.time()
import evaluator
print(f"Imported evaluator: {time.time() - t0:.2f}s")

t0 = time.time()
from langchain_community.vectorstores import FAISS
print(f"Imported FAISS: {time.time() - t0:.2f}s")

t0 = time.time()
from langchain_huggingface import HuggingFaceEmbeddings
print(f"Imported HuggingFaceEmbeddings class: {time.time() - t0:.2f}s")

t0 = time.time()
try:
    print("Loading HuggingFaceEmbeddings model 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"Loaded HuggingFaceEmbeddings model: {time.time() - t0:.2f}s")
except Exception as e:
    print(f"Failed to load HuggingFaceEmbeddings: {e}")

t0 = time.time()
if os.path.exists("faiss_index"):
    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS local: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
else:
    print("faiss_index directory not found")

t0 = time.time()
try:
    database.init_db()
    print(f"Initialized database: {time.time() - t0:.2f}s")
except Exception as e:
    print(f"Failed to initialize database: {e}")
