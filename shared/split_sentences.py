import spacy

nlp = spacy.load("en_core_web_sm")


def split_sentences(text):
    parsed = nlp(text)
    sentences = [sentence.text for sentence in parsed.sents]
    return sentences
