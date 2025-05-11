import numpy as np
from numpy.linalg import norm

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    magnitude1 = norm(embedding1)
    magnitude2 = norm(embedding2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def compare_faces(uploaded_embedding, stored_embedding, threshold=0.8):
    similarity = cosine_similarity(uploaded_embedding, stored_embedding)
    return similarity

def compare_face(uploaded_embedding, uploaded_embeddings, threshold=0.8):
    similarity = cosine_similarity(uploaded_embedding, uploaded_embeddings)
    return similarity