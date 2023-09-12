from shared.embedding import create_text_embedding
from sklearn.metrics.pairwise import cosine_similarity


def cos_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity


def is_true_embedding(question: str, answer: str, context: str):
    qa_embedding = create_text_embedding(question + " " + answer)
    context_embedding = create_text_embedding(context)

    similarity = cos_similarity(qa_embedding, context_embedding)

    return similarity


def is_true_embedding_qa_negated(question: str, answer: str, context: str):
    question_embedding = create_text_embedding(question)
    qa_embedding = create_text_embedding(question + " " + answer)
    context_embedding = create_text_embedding(context)

    similarity_with_answer = cos_similarity(qa_embedding, context_embedding)
    similarity_without_answer = cos_similarity(question_embedding, context_embedding)

    return 1 if similarity_with_answer > similarity_without_answer else 0
