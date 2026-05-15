import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

def create_vector_store(chunks):

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, embeddings

def search_query(query, chunks, index, top_k=5):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for i in indices[0]:
        results.append({
            "chunk_id": int(i),
            "text": chunks[i]
        })

    return results