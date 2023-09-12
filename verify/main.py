from shared.custom_types import DetailedQuestionAnswerPair
from classes.Validator import Validator
from context.cohere import find_context
from concurrent.futures import ThreadPoolExecutor
from is_relevant_functions.llm.main import is_relevant_llm


def determine_relevance(question, answer, context):
    return True if is_relevant_llm(question, answer, context) else False


def compute_best_prediction(question, answer, context, validator):
    prediction = validator.predict(question, answer, context["text"])
    return prediction, context


def verify(query: DetailedQuestionAnswerPair, validator: Validator, top_n=8):
    general_contexts = find_context(query.question + " " + query.answer, top_n // 2)
    for index, answer_sentence in enumerate(query.answer_sentences):
        sentence_contexts = find_context(
            query.question + " " + answer_sentence, top_n // 2
        )
        contexts = general_contexts + sentence_contexts

        best_prediction = None
        best_context = None

        with ThreadPoolExecutor() as executor:
            for prediction, context in executor.map(
                lambda ctx: compute_best_prediction(
                    query.question, answer_sentence, ctx, validator
                ),
                contexts,
            ):
                if (
                    best_prediction is None
                    or prediction["probability"] > best_prediction["probability"]
                ):
                    best_prediction = prediction
                    best_context = context

        result = {
            "id": index,
            "text": answer_sentence,
            "truth": best_prediction,
            "context": best_context,
            "relevant": determine_relevance(
                query.question, answer_sentence, best_context["text"]
            ),
        }
        yield result
