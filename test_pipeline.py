
from data.dataloader import NewsGroupDataset
from src.embeddings.embedder import EmbeddingModel
from src.vector_store.chroma_store import ChromaVectorStore


from src.clustering.fuzzy_cluster import FuzzyCluster


def main():

    dataset_path = "data/mini_newsgroups/mini_newsgroups"

    dataset = NewsGroupDataset(dataset_path)

    documents = dataset.load_dataset()

    print("Total documents:", len(documents))

    embedder = EmbeddingModel()

    embeddings = embedder.embed_batch(documents[:2000])

    vector_db = ChromaVectorStore()

    vector_db.add_documents(documents[:2000], embeddings)

    # CLUSTERING

    cluster_model = FuzzyCluster(n_clusters=10)

    cluster_probs = cluster_model.fit(embeddings)

    print("Cluster probability shape:", cluster_probs.shape)

    boundary_docs = cluster_model.get_boundary_documents()

    print("Boundary documents:", len(boundary_docs))

    cluster_model.visualize_clusters(embeddings)


if __name__ == "__main__":
    main()


