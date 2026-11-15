import json

from pathlib import Path
from src.rag_pipeline import RAGPipeline


# Load evaluation dataset
dataset_path = Path("data/evaluation/evaluation_dataset.json")

with open(dataset_path, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)


# Initialize RAG pipeline
rag = RAGPipeline(top_k=3)


# Run evaluation questions
results = []

for index, item in enumerate(evaluation_data):

    print(f"\nEvaluating question {index + 1}")

    result = rag.ask_with_context(item["question"])

    evaluation_result = {
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "answer": result["answer"],
        "contexts": result["contexts"]
    }

    results.append(evaluation_result)

    print("\nQuestion:", item["question"])
    print("\nGenerated Answer:", result["answer"])


# Save evaluation results
output_path = Path("data/evaluation/evaluation_results.json")

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)


print("\nEvaluation completed!")
print("Results saved to:", output_path)