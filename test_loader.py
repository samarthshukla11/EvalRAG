from src.ingestion.loader import load_pdf


file_path = "data/documents/employee_policy.pdf"

pages = load_pdf(file_path)

for page in pages:

    print(f"\nPage: {page['page_number']}")

    print(f"Source: {page['source']}")

    print(page["text"])