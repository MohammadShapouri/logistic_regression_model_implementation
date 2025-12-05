import numpy as np
from mnist import MnistDataManager
from metrics import BinaryMetrics

class BinaryLogisticRegression:
    def __init__(self, mnist_train_label, mnist_test_label, mnist_train_image, mnist_test_image, learning_rate, epochs, batch_size, metric_calculator, lambda_=0.0):
        self._mnist_train_label = mnist_train_label
        self._mnist_test_label = mnist_test_label
        self._mnist_train_image = mnist_train_image
        self._mnist_test_image = mnist_test_image
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.weight_matrix = None
        self.bias = None
        self.metric_calculator = metric_calculator

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _generate_binary_labels(self, labels):
        return (labels == 6).astype(np.float32).reshape(-1, 1)

    def _binary_cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-12
        batch_size = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)) / batch_size
        if self.weight_matrix is not None and self.lambda_ > 0:
            loss += (self.lambda_ / (2 * batch_size)) * np.sum(self.weight_matrix ** 2)
        return loss

    def _normalize_image_data(self, data):
        return data.reshape(-1, 28*28).astype(np.float32) / 255.0

    def prepare_data(self):
        self._mnist_train_image = self._normalize_image_data(self._mnist_train_image)
        self._mnist_test_image = self._normalize_image_data(self._mnist_test_image)
        self._mnist_train_label_bin = self._generate_binary_labels(self._mnist_train_label)
        self._mnist_test_label_bin = self._generate_binary_labels(self._mnist_test_label)

    def train_model(self):
        number_of_features = self._mnist_train_image.shape[1]
        self.weight_matrix = np.random.normal(0, 0.01, size=(number_of_features, 1)).astype(np.float32)
        self.bias = np.zeros((1, 1), dtype=np.float32)

        number_of_training_samples = self._mnist_train_image.shape[0]
        steps_per_epoch = number_of_training_samples // self.batch_size

        for epoch in range(self.epochs):
            perm = np.random.permutation(number_of_training_samples)
            shuffled_images = self._mnist_train_image[perm]
            shuffled_labels = self._mnist_train_label_bin[perm]

            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                start = step * self.batch_size
                end = start + self.batch_size
                batch_images = shuffled_images[start:end]
                batch_labels = shuffled_labels[start:end]

                logits = batch_images.dot(self.weight_matrix) + self.bias
                probs = self._sigmoid(logits)

                loss = self._binary_cross_entropy_loss(probs, batch_labels)
                epoch_loss += loss

                dlogits = (probs - batch_labels) / batch_images.shape[0]
                dW = batch_images.T.dot(dlogits)
                db = np.sum(dlogits, axis=0, keepdims=True)

                if self.lambda_ > 0:
                    dW += (self.lambda_ / batch_images.shape[0]) * self.weight_matrix

                self.weight_matrix -= self.learning_rate * dW
                self.bias -= self.learning_rate * db

            epoch_loss /= steps_per_epoch

            train_probs = self._sigmoid(self._mnist_train_image.dot(self.weight_matrix) + self.bias)
            y_pred_train = (train_probs >= 0.5).astype(np.int32)
            
            # Using the custom metrics module
            acc, _, _, _ = self.metric_calculator.calculate_report(self._mnist_train_label_bin, y_pred_train)
            
            print(f"Epoch {epoch+1}/{self.epochs}  Loss={epoch_loss:.4f}  TrainAcc={acc:.4f}")

        test_probs = self._sigmoid(self._mnist_test_image.dot(self.weight_matrix) + self.bias)
        y_pred_test = (test_probs >= 0.5).astype(np.int32)
        
        # Using the custom metrics module for final report
        acc, prec, rec, f1 = self.metric_calculator.calculate_report(self._mnist_test_label_bin, y_pred_test)
        
        print("\n" + "="*40)
        print("     FINAL TEST RESULTS")
        print("="*40)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("="*40)

if __name__ == "__main__":
    dm = MnistDataManager()
    dm.save_mnist()
    mnist_train_image, mnist_train_label, mnist_test_image, mnist_test_label = dm.load_data()
    learning_rate = 0.1
    epochs = 10
    batch_size = 256
    lambda_ = 1e-4
    lr = BinaryLogisticRegression(
        mnist_train_label, mnist_test_label, mnist_train_image, mnist_test_image,
        learning_rate, epochs, batch_size, BinaryMetrics, lambda_
    )
    lr.prepare_data()
    lr.train_model()