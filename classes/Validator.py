import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from shared.custom_types import EVALUATION_TYPE
from shared.utils import create_ground_truth_df
from is_true_functions.all import prediction_funcs_dict


"""
class Validator:
    This class is used to train and use a model to validate the truthfulness of a sentence given a context.

    Attributes:
        id: The id of the validator.
        evaluation_type: The type of evaluation to perform.
        prediction_funcs: The prediction functions to use.
        model: The model to use.
        threshold: The threshold to use.

    Methods:
        train: Trains the model.
        save: Saves the model to a file.
        load: Loads the model from a file.
        predict: Predicts the truthfulness of a sentence given a context.
        test: Tests the model on a test set.
        visualize: Visualizes the coefficients of the model.
"""


class Validator:
    def __init__(
        self,
        id,
    ):
        self.id = id
        self.evaluation_type = None
        self.prediction_funcs = None
        self.model = None
        self.threshold = None

    def train(
        self,
        prediction_funcs,
        evaluation_type: EVALUATION_TYPE = EVALUATION_TYPE.IS_TRUE,
        ground_truth_file="data/training_data.csv",
    ):
        self.prediction_funcs = [
            prediction_func["func"]
            for prediction_func in prediction_funcs.values()
            if not prediction_func["ignore"]
        ]
        self.evaluation_type = evaluation_type
        ground_truth = create_ground_truth_df(
            evaluation_type=evaluation_type,
            ground_truth_file=ground_truth_file,
        )

        training_data = pd.DataFrame().assign(
            truth=ground_truth["truth"],
        )

        for index, row in tqdm(ground_truth.iterrows()):
            question = row["question"]
            answer = row["answer"]
            context = row["context"]

            for prediction_func in self.prediction_funcs:
                prediction = prediction_func(question, answer, context)
                training_data.loc[index, prediction_func.__name__] = prediction

        X = training_data[
            [prediction_func.__name__ for prediction_func in self.prediction_funcs]
        ]
        y = training_data["truth"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Train logistic regression
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_train, y_train)

        # Predict probabilities
        y_probs = lr.predict_proba(X_test)[:, 1]

        # Find the threshold that gives the maximum F1 score
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0, 1.01, 0.01):  # testing thresholds between 0 and 1
            y_pred = (y_probs > threshold).astype(int)
            score = f1_score(y_test, y_pred)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        print(f"Best Threshold: {best_threshold:.2f}")

        # Make predictions using the best threshold
        y_pred = (y_probs > best_threshold).astype(int)

        # Evaluate the model (with best threshold)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

        self.model = lr
        self.threshold = best_threshold

    def save(self):
        data_to_save = {
            "model": self.model,
            "feature_names": [
                prediction_func.__name__ for prediction_func in self.prediction_funcs
            ],
            "evaluation_type": self.evaluation_type.value,
            "threshold": self.threshold,
        }

        with open(f"models/{self.id}.pkl", "wb") as file:
            pickle.dump(data_to_save, file)

    def load(self, prediction_funcs_dict=prediction_funcs_dict):
        with open(f"models/{self.id}.pkl", "rb") as file:
            data = pickle.load(file)
            self.model = data["model"]
            self.prediction_funcs = [
                prediction_funcs_dict[feature_name]["func"]
                for feature_name in data["feature_names"]
            ]
            self.evaluation_type = EVALUATION_TYPE(data["evaluation_type"])
            self.threshold = data["threshold"]

    def predict(self, question: str, answer: str, context: str):
        if self.model is None:
            raise Exception("Model not trained")

        prediction_data = pd.DataFrame()
        prediction_funcs = {}

        for prediction_func in self.prediction_funcs:
            prediction = prediction_func(question, answer, context)
            prediction_funcs[prediction_func.__name__] = {
                "prediction": prediction,
                "description": prediction_funcs_dict[prediction_func.__name__][
                    "description"
                ],
            }
            prediction_data.loc[0, prediction_func.__name__] = prediction

        X = prediction_data[
            [prediction_func.__name__ for prediction_func in self.prediction_funcs]
        ]

        prediction_prob = self.model.predict_proba(X)[0, 1]
        prediction = (prediction_prob > self.threshold).astype(int)

        return {
            "isValid": prediction,
            "probability": prediction_prob,
            "justification": prediction_funcs,
        }

    def test(
        self,
        ground_truth_file="data/training_data.csv",
    ):
        ground_truth = create_ground_truth_df(
            evaluation_type=self.evaluation_type,
            ground_truth_file=ground_truth_file,
        )
        truth = ground_truth["truth"]

        predictions = []
        for _, row in tqdm(ground_truth.iterrows()):
            question = row["question"]
            answer = row["answer"]
            context = row["context"]
            predictions.append(self.predict(question, answer, context).get("isValid"))

        accuracy = accuracy_score(truth, predictions)
        f1 = f1_score(truth, predictions)
        precision = precision_score(truth, predictions)
        recall = recall_score(truth, predictions)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def visualize(self):
        if self.model is None:
            raise Exception("Model not trained")

        # Get the coefficients of the model
        coefficients = self.model.coef_[0]

        # Get the names of the features
        feature_names = [
            prediction_func.__name__ for prediction_func in self.prediction_funcs
        ]

        # Create a dataframe with the coefficients and feature names
        df = pd.DataFrame(
            {"coefficients": coefficients, "feature_names": feature_names}
        )

        # Sort the dataframe by the absolute value of the coefficients
        df = df.reindex(df.coefficients.abs().sort_values(ascending=False).index)

        # Plot the coefficients
        fig = px.bar(
            df,
            x="feature_names",
            y="coefficients",
            color="coefficients",
            color_continuous_scale="RdBu",
            title="Coefficients of the logistic regression model",
        )

        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Coefficient",
            coloraxis_colorbar=dict(
                title="Coefficient",
                tickvals=[-1, 0, 1],
                ticktext=["Negative", "Zero", "Positive"],
            ),
        )

        fig.show()
