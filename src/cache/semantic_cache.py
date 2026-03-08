import numpy as np


class SemanticCache:

    def __init__(self, similarity_threshold=0.85):

        self.cache = []
        self.similarity_threshold = similarity_threshold

        self.hit_count = 0
        self.miss_count = 0

    def cosine_similarity(self, v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def search(self, query_embedding):

        best_match = None
        best_score = 0

        for entry in self.cache:

            score = self.cosine_similarity(query_embedding, entry["embedding"])

            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.similarity_threshold:

            self.hit_count += 1

            return {
                "cache_hit": True,
                "matched_query": best_match["query"],
                "similarity_score": float(best_score),
                "result": best_match["result"],
                "cluster": best_match["cluster"]
            }

        self.miss_count += 1

        return None

    def add(self, query, embedding, result, cluster):

        entry = {
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        }

        self.cache.append(entry)

    def stats(self):

        total = len(self.cache)

        if total == 0:
            hit_rate = 0
        else:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count)

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0