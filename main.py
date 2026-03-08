import sys
import os
from fastapi import FastAPI

# Ensure the app can find the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.api.routes import router, initialize
from src.embeddings.embedder import EmbeddingModel
from src.vector_store.chroma_store import ChromaVectorStore
from src.cache.semantic_cache import SemanticCache
from src.clustering.fuzzy_cluster import FuzzyCluster
from src.services.query_engine import QueryEngine
from data.dataloader import NewsGroupDataset

app = FastAPI(title="Semantic Search API")

# --- INITIALIZATION ---

# 1. Initialize Vector Store first to check existing data
print("Initializing vector database...")
vector_store = ChromaVectorStore()
collection_count = vector_store.collection.count()

# 2. Load Models
print("Loading embedding model...")
embedder = EmbeddingModel()

# 3. Handle Data and Embeddings
if collection_count == 0:
    print("Vector store is empty. Loading dataset and generating embeddings...")
    dataset_path = "data/mini_newsgroups/mini_newsgroups"
    dataset = NewsGroupDataset(dataset_path)
    documents = dataset.load_dataset()
    
    embeddings = embedder.embed_batch(documents)
    vector_store.add_documents(documents, embeddings)
else:
    print(f"Vector store already contains {collection_count} documents. Loading existing data...")
    # Fetch existing embeddings from Chroma for clustering
    stored_data = vector_store.collection.get(include=['embeddings'])
    embeddings = stored_data['embeddings']

# 4. Clustering (Always fit to ensure model state is ready)
print("Training fuzzy clustering model...")
cluster_model = FuzzyCluster(n_clusters=10)
cluster_model.fit(embeddings)

# 5. Cache and Engine
print("Initializing semantic cache...")
cache = SemanticCache(similarity_threshold=0.85)

print("Initializing query engine...")
engine = QueryEngine(
    embedder=embedder,
    vector_db=vector_store,
    cluster_model=cluster_model,
    cache=cache
)

# 6. Setup Routes
initialize(engine, cache)
app.include_router(router)

print("System ready.")