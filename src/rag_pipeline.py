from src.embeddings.embedder import (
    load_embedding_model,
    generate_embeddings
)

from src.vectorstore.chroma_store import (
    create_vector_store,
    search_chunks
)

from src.generation.gemini_client import (
    create_gemini_client,
    generate_answer
)


class RAGPipeline:

    def __init__(self, top_k=3):

        self.top_k = top_k

        self.embedding_model = load_embedding_model()

        self.collection = create_vector_store()

        self.gemini_client = create_gemini_client()

    def ask_with_context(self, question):

        # Generate question embedding
        query_embedding = generate_embeddings(
            [question],
            self.embedding_model
        )[0]

        # Retrieve relevant chunks
        results = search_chunks(
            self.collection,
            query_embedding,
            top_k=self.top_k
        )

        retrieved_chunks = results["documents"][0]

        # Generate answer
        answer = generate_answer(
            self.gemini_client,
            question,
            retrieved_chunks
        )

        return {
            "question": question,
            "answer": answer,
            "contexts": retrieved_chunks
        }

    def ask(self, question):

        result = self.ask_with_context(question)

        return result["answer"]