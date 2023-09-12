from classes.Validator import Validator
from shared.custom_types import EVALUATION_TYPE
from shared.utils import increment_version
from is_true_functions.all import prediction_funcs_dict
import os
import pickle


def load_validator_scores():
    try:
        with open("data/validator_scores.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_validator_scores(validator_scores):
    with open("data/validator_scores.pkl", "wb") as f:
        pickle.dump(validator_scores, f)


def update_validator_score(validator: Validator):
    validator_scores = load_validator_scores()
    results = validator.test()
    validator_scores[validator.id] = results["f1"]
    save_validator_scores(validator_scores)


def update_validator():
    version = increment_version()

    validator = Validator(id=f"validator_all_v{version}")
    validator.train(
        prediction_funcs=prediction_funcs_dict,
        evaluation_type=EVALUATION_TYPE.IS_TRUE,
    )
    validator.save()
    update_validator_score(validator)


def get_best_validator():
    validator_scores = load_validator_scores()
    best_validator = max(validator_scores, key=validator_scores.get)
    return best_validator
