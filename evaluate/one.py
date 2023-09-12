from shared.custom_types import EVALUATION_TYPE
from classes.EvaluateOneResponse import EvaluateOneResponse
from shared.utils import create_ground_truth_df
from tqdm import tqdm


def evaluate_one_method(
    name: str,
    prediction_func,
    evaluation_type: EVALUATION_TYPE = EVALUATION_TYPE.IS_TRUE,
    limit=None,
    ground_truth_file="data/training_data.csv",
    save=False,
    threshold=0,
):
    ground_truth = create_ground_truth_df(evaluation_type, ground_truth_file)
    prediction_column = (f"{name}_@{threshold}") if threshold else name

    TP, FP, TN, FN = 0, 0, 0, 0

    for index, row in tqdm(ground_truth.iterrows()):
        question = row["question"]
        answer = row["answer"]
        context = row["context"]
        truth = row["truth"]

        prediction = 1 if prediction_func(question, answer, context) > threshold else 0
        ground_truth.loc[index, prediction_column] = prediction

        if prediction == 1 and truth == 1:
            TP += 1
        elif prediction == 1 and truth == 0:
            FP += 1
        elif prediction == 0 and truth == 0:
            TN += 1
        elif prediction == 0 and truth == 1:
            FN += 1

        if limit and index >= limit:
            break

    if save:
        filename = f"data/results_{name}.csv"
        ground_truth.to_csv(filename, index=False)

    return EvaluateOneResponse(name, TP, FP, TN, FN)
