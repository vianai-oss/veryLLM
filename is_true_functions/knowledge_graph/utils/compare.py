from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from shared.embedding import create_text_embedding


def edge_to_str(edge):
    return f'{edge["from"]} {edge["relation"]} {edge["to"]}'


def find_closest_edges(source_edges, answer_edge, top_n=3):
    relation_strings = [edge_to_str(edge) for edge in source_edges]

    relation_string_to_embedding = {
        relation_string: create_text_embedding(relation_string)
        for relation_string in relation_strings
    }

    answer_edge_embedding = create_text_embedding(edge_to_str(answer_edge))

    # Convert dictionary to matrix
    relation_strings, embeddings = zip(
        *[
            (relation_string, embedding)
            for relation_string, embedding in relation_string_to_embedding.items()
            if embedding is not None
        ]
    )
    embeddings_matrix = np.vstack(embeddings)

    # Calculate cosine similarity for all at once
    similarities = cosine_similarity([answer_edge_embedding], embeddings_matrix)[0]

    # Get indices of top n similarities
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # Return top n sources with their similarity scores
    return [(relation_strings[i], similarities[i]) for i in top_indices]
