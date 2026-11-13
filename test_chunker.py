from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages


# Load PDF
file_path = "data/documents/employee_policy.pdf"

pages = load_pdf(file_path)

# Generate chunks
chunks = chunk_pages(
    pages,
    chunk_size=100,
    overlap=20
)

# Display results
print("Total chunks:", len(chunks))

for chunk in chunks:

    print("\n--------------------------")

    print("Metadata:", chunk["metadata"])

    print("Text:", chunk["text"])