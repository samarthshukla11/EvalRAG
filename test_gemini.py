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

from src.generation.gemini_client import (
    create_gemini_client,
    generate_answer
)


# Step 1: Load PDF
pages = load_pdf("data/documents/employee_policy.pdf")


# Step 2: Generate chunks
chunks = chunk_pages(
    pages,
    chunk_size=100,
    overlap=20
)


# Step 3: Load embedding model
model = load_embedding_model()


# Step 4: Generate embeddings
texts = [chunk["text"] for chunk in chunks]

embeddings = generate_embeddings(texts, model)


# Step 5: Connect to ChromaDB
collection = create_vector_store()

add_chunks(collection, chunks, embeddings)


# Step 6: Ask a question
question = "What is the purpose of this employee handbook?"


# Step 7: Retrieve relevant chunks
query_embedding = generate_embeddings(
    [question],
    model
)[0]

results = search_chunks(
    collection,
    query_embedding,
    top_k=3
)

retrieved_chunks = results["documents"][0]

#added after
print("\nRetrieved Context:")

for index, chunk in enumerate(retrieved_chunks):
    print(f"\nChunk {index + 1}:")
    print(chunk)


# Step 8: Connect to Gemini
client = create_gemini_client()


# Step 9: Generate answer
answer = generate_answer(
    client,
    question,
    retrieved_chunks
)


# Step 10: Display answer
print("\nUser Question:")
print(question)

print("\nGemini Answer:")
print(answer)