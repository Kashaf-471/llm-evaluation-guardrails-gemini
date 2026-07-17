"""Inspect what's stored in the Chroma vector DB at ./chroma_db."""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "langchain_docs"

embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

count = vector_store._collection.count()
print(f"Total chunks stored: {count}\n")

# Pull everything (ids, documents, metadata). Skip embeddings themselves —
# they're 768-dim floats, not human-readable.
data = vector_store._collection.get(include=["documents", "metadatas"])

# Group by source to see coverage per doc page
from collections import Counter
sources = Counter(m.get("source", "unknown") for m in data["metadatas"])
print("Chunks per source:")
for source, n in sorted(sources.items()):
    print(f"  {n:4d}  {source}")

print("\n--- Sample chunk (first one) ---")
if data["documents"]:
    print(f"ID: {data['ids'][0]}")
    print(f"Source: {data['metadatas'][0].get('source')}")
    print(f"Content preview:\n{data['documents'][0][:500]}")

# Optional: try a similarity search to sanity-check retrieval quality
print("\n--- Sample similarity search ---")
query = "how do I stream tool results from a subagent"
results = vector_store.similarity_search(query, k=3)
for i, doc in enumerate(results, 1):
    print(f"\n[{i}] {doc.metadata.get('source')}")
    print(doc.page_content[:200])