from src.rag_pipeline import RAGPipeline


# Initialize RAG pipeline
rag = RAGPipeline(top_k=3)


# Ask a question
question = "What benefits are provided to employees?"

answer = rag.ask(question)


# Display answer
print("\nQuestion:")
print(question)

print("\nGenerated Answer:")
print(answer)