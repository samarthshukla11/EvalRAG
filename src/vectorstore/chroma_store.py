import chromadb


def create_vector_store():

    # Create a persistent ChromaDB client
    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    # Create or retrieve a collection
    collection = client.get_or_create_collection(
        name="employee_policy",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def add_chunks(collection, chunks, embeddings):

    documents = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):

        documents.append(chunk["text"])

        metadatas.append(chunk["metadata"])

        ids.append(f"chunk_{index}")

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print(f"Successfully stored {len(chunks)} chunks!")


def search_chunks(collection, query_embedding, top_k=3):

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results