import matplotlib.pyplot as plt
from classes.EvaluateOneResponse import EvaluateOneResponse

"""
class EvaluateManyResponse:
    This class is used to store the results of multiple evaluations.
    It can be used to compare the results of different models or different
    thresholds for the same model.

    Attributes:
        name: The name of the model or evaluation.
        evaluations: A dictionary of threshold -> EvaluateOneResponse.
            These are the evaluations to compare.

    Methods:
        add: Adds an evaluation to the dictionary.
        find_best_threshold: Returns the threshold with the highest F1 score.
            If there are multiple thresholds with the same F1 score, the first
            one is returned.
        best: Returns the evaluation with the highest F1 score.
            If there are multiple evaluations with the same F1 score, the first
            one is returned.
        visualize: Creates and shows a matplotlib.pyplot object with the
            precision-recall curve for the evaluations.
        __str__: Returns a string representation of the object.
        

"""


class EvaluateManyResponse:
    def __init__(self, name):
        self.name = name
        self.evaluations = {}

    def add(self, threshold, summary: EvaluateOneResponse):
        self.evaluations[threshold] = summary

    def find_best_threshold(self):
        best_threshold = None
        best_f1 = 0
        for threshold, summary in self.evaluations.items():
            f1 = summary.f1()
            if f1 is None:
                continue
            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
        return best_threshold

    def best(self):
        best_threshold = self.find_best_threshold()
        if best_threshold is None:
            return None
        return self.evaluations[best_threshold]

    def visualize(self, show_best_threshold=False):
        print(self)
        if show_best_threshold:
            best_threshold = self.find_best_threshold()
            if best_threshold is None:
                print("No best threshold found")
            else:
                print("Best Threshold:", best_threshold)
        thresholds = []
        precisions = []
        recalls = []

        for threshold, summary in self.evaluations.items():
            precision = summary.precision()
            recall = summary.recall()

            if precision is None or recall is None:
                continue

            thresholds.append(threshold)
            precisions.append(precision)
            recalls.append(recall)

        plt.figure(figsize=(16, 8))
        plt.plot(recalls, precisions, marker="o")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve for " + self.name)
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)

        for i, threshold in enumerate(thresholds):
            plt.annotate(
                f"{threshold:.2f}",
                (recalls[i], precisions[i]),
                textcoords="offset points",
                xytext=(-5, 5),
                ha="center",
            )

        plt.show()

    def __str__(self):
        return f"EvaluateResponse for {self.name}:\n" + "\n".join(
            [
                f"{threshold}: {summary}"
                for threshold, summary in self.evaluations.items()
            ]
        )
