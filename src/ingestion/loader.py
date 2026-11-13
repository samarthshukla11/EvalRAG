import pymupdf
from pathlib import Path


def load_pdf(file_path):

    pdf_path = Path(file_path)

    # Check whether file exists
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    # Check file extension
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    extracted_pages = []

    try:
        document = pymupdf.open(pdf_path)

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "page_number": page_number,
                    "text": text,
                    "source": pdf_path.name
                })

        document.close()

    except Exception as error:
        raise RuntimeError(
            f"Failed to process PDF: {error}"
        ) from error

    if not extracted_pages:
        raise ValueError(
            "No extractable text found in PDF"
        )

    return extracted_pages