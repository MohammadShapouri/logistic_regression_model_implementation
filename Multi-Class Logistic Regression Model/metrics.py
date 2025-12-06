import numpy as np

class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculates simple accuracy: (Correct / Total)
        """
        return np.mean(y_true == y_pred)

    @staticmethod
    def print_classification_report(y_true, y_pred, num_classes):
        """
        Calculates and prints Precision, Recall, and F1-Score for each class.
        Logic:
        - TP (True Positive): Predicted X, Actual X
        - FP (False Positive): Predicted X, Actual NOT X
        - FN (False Negative): Predicted NOT X, Actual X
        """
        print("\nClassification report (per-class precision/recall/f1):")
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 45)

        for i in range(num_classes):
            # True Positive: Predicted i AND Actual i
            tp = np.sum((y_pred == i) & (y_true == i))
            # False Positive: Predicted i BUT Actual is NOT i
            fp = np.sum((y_pred == i) & (y_true != i))
            # False Negative: Predicted NOT i BUT Actual is i
            fn = np.sum((y_pred != i) & (y_true == i))

            # Avoid division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"{i:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
