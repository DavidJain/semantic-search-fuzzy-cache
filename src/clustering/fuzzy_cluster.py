import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FuzzyCluster:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42
        )
        self.cluster_probabilities = None

    def fit(self, embeddings):
        print("Training Gaussian Mixture clustering...")
        self.model.fit(embeddings)
        self.cluster_probabilities = self.model.predict_proba(embeddings)
        return self.cluster_probabilities

    def get_dominant_cluster(self):
        if self.cluster_probabilities is None:
            raise ValueError("Model must be fitted before getting clusters.")
        return np.argmax(self.cluster_probabilities, axis=1)

    def get_boundary_documents(self, threshold=0.4):
        """
        Identifies documents that belong to multiple clusters 
        based on the probability gap between the top two clusters.
        """
        if self.cluster_probabilities is None:
            return []
            
        boundary_docs = []
        for i, probs in enumerate(self.cluster_probabilities):
            sorted_probs = sorted(probs, reverse=True)
            # If the difference between top 2 clusters is small, it's a boundary doc
            if abs(sorted_probs[0] - sorted_probs[1]) < threshold:
                boundary_docs.append(i)
        return boundary_docs

    def visualize_clusters(self, embeddings):
        print("Reducing dimensions for visualization...")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        labels = self.get_dominant_cluster()

        plt.figure(figsize=(10, 7))
        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels,
            cmap="tab20",
            s=10
        )
        plt.title("Document Cluster Visualization")
        plt.colorbar(label='Cluster ID')
        plt.show()

    def find_optimal_clusters(self, embeddings):
        """
        Uses Bayesian Information Criterion (BIC) to find the best k.
        Lower BIC score indicates a better model fit.
        """
        scores = []
        k_values = range(5, 20)

        for k in k_values:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            scores.append(bic)

        # The best k is the one with the lowest BIC score
        best_k = k_values[np.argmin(scores)]
        print(f"Optimal number of clusters found: {best_k}")
        return best_k