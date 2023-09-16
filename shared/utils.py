import pandas as pd
import numpy as np
from shared.custom_types import EVALUATION_TYPE


def load_csv_to_df(csv_path):
    return pd.read_csv(csv_path)


def save_df_to_csv(df, csv_path):
    df.to_csv(csv_path, index=False)


def create_ground_truth_df(
    evaluation_type=EVALUATION_TYPE.IS_TRUE,
    ground_truth_file="data/training_data.csv",
):
    ground_truth_full = load_csv_to_df(ground_truth_file)
    truth_column = evaluation_type.value

    question = ground_truth_full["question"]
    answer = ground_truth_full["answer"]
    context = ground_truth_full["context"]
    truth = ground_truth_full[truth_column]
    return pd.DataFrame().assign(
        question=question, answer=answer, context=context, truth=truth
    )


def int64_to_int(obj):
    if isinstance(obj, dict):
        return {k: int64_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [int64_to_int(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    else:
        return obj


def get_current_version():
    with open("data/version.txt", "r") as f:
        version = f.read().strip()
    return int(version)


def increment_version():
    version = get_current_version()
    version += 1
    with open("data/version.txt", "w") as f:
        f.write(str(version))
    return version
