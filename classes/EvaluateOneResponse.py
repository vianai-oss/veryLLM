import numpy as np
import matplotlib.pyplot as plt
import itertools

"""
class EvaluateOneResponse:
    This class is used to store the results of a single evaluation.

    Attributes:
        name: The name of the model or evaluation.
        TP: The number of true positives.
        FP: The number of false positives.
        TN: The number of true negatives.
        FN: The number of false negatives.

    Methods:
        accuracy: Returns the accuracy of the evaluation.
        precision: Returns the precision of the evaluation.
        recall: Returns the recall of the evaluation.
        f1: Returns the F1 score of the evaluation.
        to_dict: Returns a dictionary representation of the evaluation.
        visualize: Plots the confusion matrix for the evaluation.
        __str__: Returns a string representation of the evaluation.

"""


class EvaluateOneResponse:
    def __init__(self, name, TP, FP, TN, FN):
        self.name = name
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN

    def accuracy(self):
        denominator = self.TP + self.TN + self.FP + self.FN
        if denominator == 0:
            return None
        return (self.TP + self.TN) / denominator

    def precision(self):
        denominator = self.TP + self.FP
        if denominator == 0:
            return None
        return self.TP / denominator

    def recall(self):
        denominator = self.TP + self.FN
        if denominator == 0:
            return None
        return self.TP / denominator

    def f1(self):
        if self.precision() is None or self.recall() is None:
            return None
        if self.precision() + self.recall() == 0:
            return None
        return (
            2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        )

    def to_dict(self):
        return {
            "name": self.name,
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
        }

    def visualize(self):
        print(self)
        cm = np.array([[self.TP, self.FP], [self.FN, self.TN]])
        classes = ["True", "False"]
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("Predicted Labels")
        plt.xlabel("Actual Labels")
        plt.tight_layout()
        plt.show()

    def __str__(self):
        return f"""
        {self.name}:
        TP: {self.TP}
        FP: {self.FP}
        TN: {self.TN}
        FN: {self.FN}
        Accuracy: {self.accuracy()}
        Precision: {self.precision()}
        Recall: {self.recall()}
        F1: {self.f1()}
        """
