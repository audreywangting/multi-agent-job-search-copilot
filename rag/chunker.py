def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap must be 0 or greater.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []
    start = 0
    chunk_id = 0
    clean_text = text.strip()

    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append({"chunk_id": chunk_id, "text": chunk})
            chunk_id += 1
        if end >= len(clean_text):
            break
        start = end - overlap

    return chunks
