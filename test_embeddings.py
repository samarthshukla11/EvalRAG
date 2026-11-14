from src.embeddings.embedder import (
    load_embedding_model,
    generate_embeddings
)


# Sample sentences
texts = [
    "Employees receive annual leave.",
    "Employees are entitled to vacation days."
]


# Load model
model = load_embedding_model()


# Generate embeddings
embeddings = generate_embeddings(texts, model)


# Display results
print("Embedding shape:", embeddings.shape)

print("\nFirst sentence embedding:")
print(embeddings[0])

print("\nSecond sentence embedding:")
print(embeddings[1])