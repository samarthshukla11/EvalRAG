def chunk_pages(pages, chunk_size=100, overlap=20):

    # Validate chunking parameters
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size"
        )

    chunks = []

    for page in pages:

        text = page["text"]

        # Convert text into individual words
        words = text.split()

        start = 0
        chunk_index = 1

        while start < len(words):

            # Determine the ending position
            end = min(start + chunk_size, len(words))

            # Extract words for the current chunk
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index
                }
            })

            chunk_index += 1

            # Stop when all words are processed
            if end == len(words):
                break

            # Move forward while maintaining overlap
            start = end - overlap

    return chunks
