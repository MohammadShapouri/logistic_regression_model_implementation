import numpy as np

class BinaryMetrics:
    @staticmethod
    def calculate_report(y_true, y_pred):
        """
        Calculates Accuracy, Precision, Recall, and F1 Score.
        Returns them as a tuple.
        """
        # Ensure inputs are flat 1D arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Calculate Confusion Matrix Components
        # TP: True = 1, Pred = 1
        tp = np.sum((y_true == 1) & (y_pred == 1))
        # TN: True = 0, Pred = 0
        tn = np.sum((y_true == 0) & (y_pred == 0))
        # FP: True = 0, Pred = 1
        fp = np.sum((y_true == 0) & (y_pred == 1))
        # FN: True = 1, Pred = 0
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Epsilon to prevent division by zero
        eps = 1e-15
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        return accuracy, precision, recall, f1