# (c) 2023 Vianai Systems, Inc.
import argparse

from classes.Evaluator import Evaluator
from is_true_functions.all import prediction_funcs_dict
from is_relevant_functions.all import is_relevant_funcs_dict
from shared.custom_types import EVALUATION_TYPE

parser = argparse.ArgumentParser(description="Test your prediction function locally")

parser.add_argument(
    "-r",
    "--relevant",
    action="store_true",
    help="Whether to test an is_relevant function instead of an is_true function",
)
parser.add_argument(
    "-f",
    "--function",
    help="Name of function to test",
)
parser.add_argument(
    "-t",
    "--threshold",
    help="Threshold to use for prediction function",
)
args = vars(parser.parse_args())

func_name = args["function"]

if args["relevant"]:
    func = is_relevant_funcs_dict[func_name]
else:
    func = prediction_funcs_dict[func_name]

evalulator = Evaluator(func["func"])
evaluation_type = (
    EVALUATION_TYPE.IS_RELEVANT if args["relevant"] else EVALUATION_TYPE.IS_TRUE
)

if args["threshold"]:
    result = evalulator.evaluate_one(
        threshold=float(args["threshold"]), evaluation_type=evaluation_type
    )
    result.visualize()
else:
    if func["requires_threshold"]:
        result = evalulator.evaluate_many(evaluation_type=evaluation_type)
        result.visualize(show_best_threshold=True)
    else:
        result = evalulator.evaluate_one(evaluation_type=evaluation_type)
        result.visualize()
