import os


EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def get_embedding(text: str, client) -> list[float]:
    print(f"Using embedding model before API call: {EMBEDDING_MODEL}")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[dict], client) -> list[dict]:
    embedded_chunks = []
    for chunk in chunks:
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": get_embedding(chunk["text"], client),
            }
        )
    return embedded_chunks
