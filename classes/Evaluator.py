from shared.custom_types import EVALUATION_TYPE
from evaluate.one import evaluate_one_method
from evaluate.many import evaluate_many_method


"""
class Evaluator:
    This class is used to evaluate a prediction function.

    Attributes:
        name: The name of the model or evaluation.
        prediction_func: The prediction function to evaluate.
        ground_truth_file: The path to the ground truth file.

    Methods:
        evaluate_many: Evaluates the prediction function for multiple thresholds.
        evaluate_one: Evaluates the prediction function for a single threshold.
"""


class Evaluator:
    def __init__(
        self,
        prediction_func,
        ground_truth_file="data/training_data.csv",
    ):
        self.name = prediction_func.__name__
        self.prediction_func = prediction_func
        self.ground_truth_file = ground_truth_file

    def evaluate_many(
        self,
        evaluation_type: EVALUATION_TYPE = EVALUATION_TYPE.IS_TRUE,
        threshold_vals=[i / 100 for i in range(0, 101)],
    ):
        min_threshold = min(threshold_vals)
        max_threshold = max(threshold_vals)
        step = threshold_vals[1] - threshold_vals[0]
        print(
            f"Evaluating {self.name} with thresholds {min_threshold} to {max_threshold} in steps of {step}"
        )
        evalution = evaluate_many_method(
            name=self.name,
            prediction_func=self.prediction_func,
            evaluation_type=evaluation_type,
            ground_truth_file=self.ground_truth_file,
            thresholds=threshold_vals,
        )
        print()
        return evalution

    def evaluate_one(
        self,
        evaluation_type: EVALUATION_TYPE = EVALUATION_TYPE.IS_TRUE,
        threshold=0.5,
        save=False,
        limit=None,
    ):
        print(f"Evaluating {self.name} with threshold {threshold}")
        evalution = evaluate_one_method(
            name=self.name,
            prediction_func=self.prediction_func,
            evaluation_type=evaluation_type,
            limit=limit,
            ground_truth_file=self.ground_truth_file,
            save=save,
            threshold=threshold,
        )
        print()
        return evalution
