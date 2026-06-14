from rag.embeddings import get_embedding


def cosine_similarity(a, b) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve_top_k(
    job_description: str,
    embedded_chunks: list[dict],
    client,
    k: int = 5,
) -> list[dict]:
    job_embedding = get_embedding(job_description, client)
    scored_chunks = []

    for chunk in embedded_chunks:
        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "similarity": cosine_similarity(job_embedding, chunk["embedding"]),
            }
        )

    return sorted(
        scored_chunks,
        key=lambda chunk: chunk["similarity"],
        reverse=True,
    )[:k]
