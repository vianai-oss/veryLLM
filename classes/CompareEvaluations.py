import matplotlib.pyplot as plt

from classes.EvaluateManyResponse import EvaluateManyResponse
from classes.EvaluateOneResponse import EvaluateOneResponse

"""
class CompareEvaluations:
    This class is used to compare the results of multiple evaluations.
    It can be used to compare the results of different models or different
    thresholds for the same model.

    Attributes:
        evaluations: A list of EvaluateManyResponse or EvaluateOneResponse objects.
            These are the evaluations to compare.

    Methods:
        best: Returns the best evaluation. If there are multiple evaluations
            with the same F1 score, the first one is returned.
        create_image: Creates a matplotlib.pyplot object with the precision-recall
            curve for the evaluations.
        visualize: Creates and shows a matplotlib.pyplot object with the
            precision-recall curve for the evaluations.
        save: Creates and saves a matplotlib.pyplot object with the
            precision-recall curve for the evaluations.

"""


class CompareEvaluations:
    def __init__(self, evaluations: [EvaluateManyResponse or EvaluateOneResponse]):
        self.evaluations = evaluations

    def best(self):
        best = None
        for evaluation in self.evaluations:
            if isinstance(evaluation, EvaluateManyResponse):
                for _, summary in evaluation.evaluations.items():
                    if best is None or (
                        summary.f1() is not None and summary.f1() > best.f1()
                    ):
                        best = summary

            else:
                if best is None or (
                    summary.f1() is not None and summary.f1() > best.f1()
                ):
                    best = evaluation
        return best

    def create_image(self):
        plt.figure(figsize=(16, 8))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)

        for evaluation in self.evaluations:
            if isinstance(evaluation, EvaluateManyResponse):
                thresholds = []
                precisions = []
                recalls = []

                for threshold, summary in evaluation.evaluations.items():
                    precision = summary.precision()
                    recall = summary.recall()

                    if precision is None or recall is None:
                        continue

                    thresholds.append(threshold)
                    precisions.append(precision)
                    recalls.append(recall)

                plt.plot(recalls, precisions, marker="o", label=evaluation.name)

                for i, threshold in enumerate(thresholds):
                    plt.annotate(
                        f"{threshold:.2f}",
                        (recalls[i], precisions[i]),
                        textcoords="offset points",
                        xytext=(-5, 5),
                        ha="center",
                    )
            else:
                plt.scatter(
                    evaluation.recall(),
                    evaluation.precision(),
                    marker="o",
                    label=evaluation.name,
                )

        plt.legend()
        return plt

    def visualize(self):
        plt = self.create_image()
        plt.show()

    def save(self, path):
        plt = self.create_image()
        plt.savefig(path)
