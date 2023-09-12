from enum import Enum


class EVALUATION_TYPE(Enum):
    IS_TRUE = "truth"
    IS_INFERED = "infered"
    IS_RELEVANT = "relevant"


class QuestionAnswerPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


class DetailedQuestionAnswerPair:
    def __init__(self, question: str, answer: str, answer_sentences: list):
        self.question = question
        self.answer = answer
        self.answer_sentences = answer_sentences
