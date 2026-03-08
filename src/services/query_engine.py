class QueryEngine:
    def __init__(self, embedder, vector_db, cluster_model, cache):
        self.embedder = embedder
        self.vector_db = vector_db
        self.cluster_model = cluster_model
        self.cache = cache

    def process_query(self, query):
        # 1. Embed query
        query_embedding = self.embedder.embed_text(query)

        # 2. Check Cache
        cache_result = self.cache.search(query_embedding)
        if cache_result:
            return {
                "query": query,
                "cache_hit": True,
                "matched_query": cache_result["matched_query"],
                "similarity_score": cache_result.get("similarity_score", 0.0),
                "result": cache_result["result"],
                "dominant_cluster": cache_result["cluster"]
            }

        # 3. Search Vector DB
        results = self.vector_db.query(query_embedding)
        
        # SAFETY CHECK: Ensure results and documents exist
        if not results or not results.get("documents") or len(results["documents"][0]) == 0:
            return {
                "query": query,
                "cache_hit": False,
                "matched_query": None,
                "similarity_score": 0.0,
                "result": "No relevant documents found in the database.",
                "dominant_cluster": -1
            }

        documents = results["documents"][0]
        result_text = documents[0]

      

        # 4. Clustering Logic
        try:
            # Reshape query_embedding to (1, -1) for scikit-learn
            q_emb_reshaped = query_embedding.reshape(1, -1)
            cluster_probs = self.cluster_model.model.predict_proba(q_emb_reshaped)
            dominant_cluster = int(cluster_probs.argmax())
        except Exception as e:
            print(f"Clustering error: {e}")
            dominant_cluster = 0