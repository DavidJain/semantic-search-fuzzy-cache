from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

    def embed_text(self, text):

        return self.model.encode(text)

    def embed_batch(self, texts):

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True
        )

        return embeddings