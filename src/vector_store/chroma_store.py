import chromadb
from chromadb.config import Settings

class ChromaVectorStore:
    def __init__(self):
        # Use PersistentClient for newer versions of ChromaDB to ensure 
        # data is actually saved to the disk.
        self.client = chromadb.PersistentClient(path="chroma_db")

        self.collection = self.client.get_or_create_collection(
            name="news_embeddings"
        )

    def add_documents(self, documents, embeddings):
        """
        Adds documents only if the collection is currently empty to prevent
        duplicate ID errors on container restart.
        """
        # Check if we already have data in this persistent collection
        if self.collection.count() > 0:
            print(f"Vector store already contains {self.collection.count()} documents. Skipping insertion.")
            return

        print(f"Adding {len(documents)} new documents to the vector store...")
        ids = [str(i) for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

    def query(self, query_embedding, k=5):
        """
        Retrieves the top k most similar documents.
        """
        # Handle potential empty results gracefully
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results