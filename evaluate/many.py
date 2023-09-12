from shared.custom_types import EVALUATION_TYPE
from classes.EvaluateOneResponse import EvaluateOneResponse
from classes.EvaluateManyResponse import EvaluateManyResponse
from shared.utils import create_ground_truth_df
from tqdm import tqdm


def evaluate_many_method(
    name: str,
    prediction_func,
    evaluation_type: EVALUATION_TYPE = EVALUATION_TYPE.IS_TRUE,
    ground_truth_file="data/training_data.csv",
    thresholds=[i / 100 for i in range(0, 101)],
):
    reponse = EvaluateManyResponse(name)
    ground_truth = create_ground_truth_df(evaluation_type, ground_truth_file)
    summary_dict = {threshold: [0, 0, 0, 0] for threshold in thresholds}

    for _, row in tqdm(ground_truth.iterrows()):
        question = row["question"]
        answer = row["answer"]
        context = row["context"]
        truth = row["truth"]

        predicted_similarity = prediction_func(question, answer, context)

        for threshold in thresholds:
            TP, FP, TN, FN = summary_dict[threshold]
            prediction = 1 if predicted_similarity > threshold else 0

            if prediction == 1 and truth == 1:
                TP += 1
            elif prediction == 1 and truth == 0:
                FP += 1
            elif prediction == 0 and truth == 0:
                TN += 1
            elif prediction == 0 and truth == 1:
                FN += 1

            summary_dict[threshold] = [TP, FP, TN, FN]

    for threshold in thresholds:
        TP, FP, TN, FN = summary_dict[threshold]
        reponse.add(threshold, EvaluateOneResponse(name, TP, FP, TN, FN))

    return reponse
