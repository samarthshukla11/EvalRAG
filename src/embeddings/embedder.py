from sentence_transformers import SentenceTransformer


# Load the embedding model
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


# Convert text into embeddings
def generate_embeddings(texts, model):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings