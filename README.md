# semantic-search-fuzzy-cache
AI-powered semantic retrieval system with fuzzy clustering, vector search, and a custom semantic cache API.

# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a **lightweight semantic search system** using the **20 Newsgroups dataset**.  
It combines **vector embeddings, fuzzy clustering, and a custom semantic cache** to retrieve documents based on meaning rather than keyword matching.

This project was built as part of the **Trademarkia AI/ML Engineer assignment**.

---

# Overview

Traditional search systems rely on **keyword matching**, which fails when users ask the same question in different ways.

This system instead performs **semantic search** by converting text into vector embeddings and retrieving documents based on similarity in embedding space.

The system has three main components:

1. **Vector embeddings + vector database**
2. **Fuzzy clustering of documents**
3. **Semantic cache for query reuse**

---

# Architecture
User Query
│
▼
Query Embedding (SentenceTransformer)
│
▼
Semantic Cache
│
┌──Cache Hit─────────────┐
│ │
▼ ▼
Return Cached Result Vector Search (ChromaDB)
│
▼
Fuzzy Cluster Detection
│
▼
Return Result


---

# Dataset

The system uses the **20 Newsgroups dataset**, which contains approximately **20,000 documents across 20 topic categories**.

Example topics include:

- politics
- religion
- space
- sports
- computer hardware
- science

However, documents often contain **multiple overlapping themes**, making it suitable for **fuzzy clustering**.

Dataset source:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

---

# Dataset Preprocessing

The raw dataset contains several sources of noise such as:

- email headers
- email addresses
- URLs
- quoted replies
- excessive whitespace

These elements can distort semantic embeddings, so the preprocessing pipeline removes:

- email addresses
- URLs
- quoted reply lines
- redundant whitespace

This ensures that embeddings represent **actual semantic content** rather than metadata artifacts.

---

# Embedding Model

The system uses the **Sentence Transformers model**

This model was chosen because:

- it produces high-quality semantic embeddings
- it is lightweight and efficient
- it generates compact 384-dimensional vectors

These properties make it ideal for **fast semantic retrieval systems**.

---

# Vector Database

Embeddings are stored using **ChromaDB**, a lightweight vector database.

ChromaDB provides:

- fast similarity search
- persistent storage for embeddings
- simple local deployment

This allows efficient retrieval of documents semantically similar to a query.

---

# Fuzzy Clustering

Instead of assigning each document to a single cluster, this system performs **soft clustering using Gaussian Mixture Models (GMM)**.

Each document receives a **probability distribution across clusters**.

Example:
Document 134

Cluster 3 → 0.52
Cluster 7 → 0.31
Cluster 2 → 0.17


This reflects the real structure of the dataset where documents may belong to **multiple semantic topics**.

---

# Cluster Selection

The number of clusters was determined using the **Bayesian Information Criterion (BIC)** across multiple values of K.

The optimal cluster count was selected by minimizing the BIC score, balancing model complexity and clustering quality.

---

# Boundary Documents

Some documents lie on **cluster boundaries**, meaning they have similar probabilities across multiple clusters.

Example:


Cluster A → 0.46
Cluster B → 0.44


These boundary cases highlight **semantic overlap between topics**.

---

# Semantic Cache

Traditional caches only match **exact queries**.

This system implements a **semantic cache**, which compares query embeddings using **cosine similarity**.

If similarity exceeds a threshold (default = **0.85**), the cached result is reused.

Example:


Query 1: "space shuttle launch"
Query 2: "NASA rocket launch"


These queries share similar meaning, so the second query can reuse the cached result.

---

# Cache Threshold Exploration

The similarity threshold determines cache behaviour.

| Threshold | Behaviour |
|--------|--------|
0.75 | higher hit rate but more false matches |
0.85 | balanced performance |
0.95 | fewer hits but higher precision |

A threshold of **0.85** was selected as a balance between accuracy and efficiency.

---

# API Service

The system exposes a **FastAPI service** for querying.

---

## POST /query

Request:

```json
{
 "query": "space shuttle launch"
}
Response:

{
 "query": "...",
 "cache_hit": false,
 "matched_query": null,
 "similarity_score": 0.0,
 "result": "...",
 "dominant_cluster": 3
}
GET /cache/stats

Returns current cache statistics.

Example:

{
 "total_entries": 42,
 "hit_count": 17,
 "miss_count": 25,
 "hit_rate": 0.405
}
DELETE /cache

Clears the cache and resets statistics.

Running the Project
Create environment
python -m venv venv

Activate:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the API server:

uvicorn main:app --reload

API documentation will be available at:

http://localhost:8000/docs
Docker Deployment

Build the container:

docker build -t semantic-search .

Run the container:

docker run -p 8000:8000 semantic-search

Or using docker-compose:

docker-compose up --build
Project Structure
src/
├── api
├── cache
├── clustering
├── data_loader
├── embeddings
├── services
└── vector_store
Technologies Used

Python

FastAPI

Sentence Transformers

ChromaDB

Scikit-learn

Docker
