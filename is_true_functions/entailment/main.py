from transformers import BartForSequenceClassification, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")
model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")


def get_entailment_prob(premise, hypothesis):
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors="pt")
    logits = model(input_ids)[0]

    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_prob = probs[:, 1].item()
    return true_prob


def is_true_entailment(question: str, answer: str, context: str):
    # premise = "A mammal ( from Latin   mamma  breast  ) is a vertebrate animal of the class Mammalia ( ) ."
    # hypothesis = "No, not all mammals are vertebrates."
    premise = context
    hypothesis = question + " " + answer

    true_prob = get_entailment_prob(premise, hypothesis)

    return true_prob


def is_true_entailment_only_answer(question: str, answer: str, context: str):
    # premise = "A mammal ( from Latin   mamma  breast  ) is a vertebrate animal of the class Mammalia ( ) ."
    # hypothesis = "No, not all mammals are vertebrates."
    premise = context
    hypothesis = answer

    true_prob = get_entailment_prob(premise, hypothesis)

    return true_prob
