import pandas as pd
import pickle
from tabulate import tabulate
from classes.Evaluator import Evaluator
from is_true_functions.all import (
    prediction_funcs,
    prediction_funcs_dict,
)
from classes.CompareEvaluations import CompareEvaluations
from updates.update_validator import load_validator_scores

CSV_FILE = "data/leaderboard.csv"


def read_csv_data(csv_file=CSV_FILE):
    try:
        return pd.read_csv(csv_file)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Function", "Description", "Author", "F1 Score"])


def save_to_csv(df, csv_file=CSV_FILE):
    df = df.sort_values(by="F1 Score", ascending=False)
    df.to_csv(csv_file, index=False)
    update_readme_with_leaderboard(df)


def find_existing_functions():
    return read_csv_data()["Function"].tolist()


def filter_new_functions(prediction_funcs, existing_functions):
    return [
        func
        for func in prediction_funcs
        if func["func"].__name__ not in existing_functions
    ]


def evaluate_function(prediction_func):
    evaluator = Evaluator(prediction_func["func"])
    if prediction_func["requires_threshold"] is False:
        evaluation = evaluator.evaluate_one(threshold=0.5)
        f1_score = evaluation.f1()
    else:
        evaluation = evaluator.evaluate_many()
        f1_score = evaluation.best().f1()

    row = {
        "Function": prediction_func["func"].__name__,
        "Description": prediction_func["description"],
        "Author": prediction_func["author"],
        "F1 Score": f1_score,
    }
    return {"row": row, "evaluation": evaluation}


def get_validator_scores():
    scores = load_validator_scores()
    print(scores)
    rows = []
    for validator, score in scores.items():
        row = {
            "Function": validator,
            "Description": "All prediction functions combined (excluding ones marked as ignore)",
            "Author": "Team",
            "F1 Score": score,
        }
        rows.append(row)
    return rows


def append_and_sort_csv(results):
    df_existing = read_csv_data()
    df_new = pd.DataFrame(results)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    save_to_csv(df_combined)


def dataframe_to_markdown_table(df):
    return tabulate(df, headers="keys", tablefmt="pipe", showindex=False)


def update_readme_with_leaderboard(df, readme_file="README.md"):
    # Convert it to markdown format
    table = dataframe_to_markdown_table(df)

    # Read in the existing README
    with open(readme_file, "r") as file:
        content = file.read()

    # Locate the start and end markers
    start_marker = "<!-- LEADERBOARD_START -->"
    end_marker = "<!-- LEADERBOARD_END -->"

    start_index = content.index(start_marker) + len(start_marker)
    end_index = content.index(end_marker)

    # Replace the old table with the new one between the markers
    updated_content = content[:start_index] + "\n" + table + "\n" + content[end_index:]

    # Save the updated README back
    with open(readme_file, "w") as file:
        file.write(updated_content)


def update_readme_with_evaluations(evals, readme_file="README.md"):
    comparison = CompareEvaluations(evals)
    image_path = "data/precision_recall_curves.png"
    comparison.save(image_path)
    # put the image in the readme
    with open(readme_file, "r") as file:
        content = file.read()

    start_marker = "<!-- EVALUATIONS_START -->"
    end_marker = "<!-- EVALUATIONS_END -->"

    start_index = content.index(start_marker) + len(start_marker)
    end_index = content.index(end_marker)

    updated_content = (
        content[:start_index]
        + "\n"
        + f"![Precision Recall Curves]({image_path})"
        + "\n"
        + content[end_index:]
    )

    with open(readme_file, "w") as file:
        file.write(updated_content)


def save_evals_to_pkl(evals, pkl_file="data/evaluations.pkl"):
    with open(pkl_file, "wb") as file:
        pickle.dump(evals, file)


def load_evals_from_pkl(pkl_file="data/evaluations.pkl"):
    try:
        with open(pkl_file, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return []


def update_evals(new_evals):
    evals = load_evals_from_pkl()
    evals = [
        eval
        for eval in evals
        if eval.name not in [new_eval.name for new_eval in new_evals]
    ]
    evals.extend(new_evals)
    update_readme_with_evaluations(evals)
    save_evals_to_pkl(evals)


def manual_update_leaderboard(prediction_func_name):
    prediction_func = prediction_funcs_dict[prediction_func_name]
    result = evaluate_function(prediction_func)
    row = result["row"]
    evaluation = result["evaluation"]
    update_evals([evaluation])
    df = read_csv_data()

    if prediction_func_name in df["Function"].tolist():
        df = df[df["Function"] != prediction_func_name]

    df_combined = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_to_csv(df_combined)


def auto_update_leaderboard():
    existing_functions = find_existing_functions()
    new_functions = filter_new_functions(prediction_funcs, existing_functions)
    results = [evaluate_function(func) for func in new_functions]
    rows = [result["row"] for result in results]
    evaluations = [result["evaluation"] for result in results]
    update_evals(evaluations)
    append_and_sort_csv(rows)


def update_validator_score():
    validator_results = get_validator_scores()
    validator_names = [
        validator_results["Function"] for validator_results in validator_results
    ]
    df = read_csv_data()
    df = df[~df["Function"].isin(validator_names)]
    df_combined = pd.concat([df, pd.DataFrame(validator_results)], ignore_index=True)
    save_to_csv(df_combined)
