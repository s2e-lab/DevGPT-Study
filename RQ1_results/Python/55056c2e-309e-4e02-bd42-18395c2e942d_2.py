class MyEmbeddingModel(EmbeddingModel):
    def create_embedding(self, text: str):
        # Use your own method to create an embedding
        # Please replace with actual embedding creation code
        return my_embedding_method(text)
