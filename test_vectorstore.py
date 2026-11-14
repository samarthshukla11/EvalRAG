from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages

from src.embeddings.embedder import (
    load_embedding_model,
    generate_embeddings
)

from src.vectorstore.chroma_store import (
    create_vector_store,
    add_chunks,
    search_chunks
)


# Step 1: Load PDF
pages = load_pdf("data/documents/employee_policy.pdf")

print("Total pages:", len(pages))


# Step 2: Generate chunks
chunks = chunk_pages(
    pages,
    chunk_size=100,
    overlap=20
)

print("Total chunks:", len(chunks))


# Step 3: Load embedding model
model = load_embedding_model()


# Step 4: Generate embeddings
texts = [chunk["text"] for chunk in chunks]

embeddings = generate_embeddings(texts, model)

print("Embedding shape:", embeddings.shape)


# Step 5: Store in ChromaDB
collection = create_vector_store()

add_chunks(collection, chunks, embeddings)


# Step 6: Search
query = "What are the responsibilities of employees?"

query_embedding = generate_embeddings(
    [query],
    model
)[0]

results = search_chunks(
    collection,
    query_embedding,
    top_k=3
)


# Step 7: Display results
print("\nMost Relevant Results:")

for index, document in enumerate(results["documents"][0]):

    print(f"\nResult {index + 1}:")
    print(document)

    print("Metadata:", results["metadatas"][0][index])