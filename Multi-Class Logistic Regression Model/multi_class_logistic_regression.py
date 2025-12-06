import numpy as np
from mnist import MnistDataManager
from metrics import ClassificationMetrics

class MultiClassLogisticRegression:
    """
    # Logitic Regression calculates X dot weight_matrix + bias to generate output.
    - 1. Logistic regression (and softmax regression) is a linear model.
    - 2. Each MNIST image is (28×28) which will be flattened into one dimentional vector containing 784 one and zero, each one shows a pixel of that image.
    - 3. The model learns how important each pixel is for each digit and it will be saved in weight_matrix.
    - 4. For detecting numbers from new images, it multiply X by weight_matrix to compute a score for each digit, it generates 10
      scores, each is a weighted sum of all pixels for example for number 0 we have:
      score_0 = x1*w1_0 + x2*w2_0 + ... + x784*w784_0 = how much the image looks like "0"
      then it adds bias which shifts the scores up/down. so -> output = X·weight_matrix + bias
    - 5. Softmax converts output into probabilities: probabilities = softmax(output)
      it returns [0.01, 0.02, 0.05, 0.03, 0.01, 0.80, 0.02, 0.01, 0.03, 0.02] as output and number 5 has the highest probability do model predicts input is number 5
    """
    def __init__(self, mnist_train_label, mnist_test_label, mnist_train_image, mnist_test_image, learning_rate, epochs, batch_size, metric_calculator, lambda_=0.0):
        self._mnist_train_label = mnist_train_label
        self._mnist_test_label = mnist_test_label
        self._mnist_train_image = mnist_train_image
        self._mnist_test_image = mnist_test_image
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self._mnist_train_label_one_hot_data = None
        self._mnist_test_label_one_hot_data = None
        # 0 to 9 means 10 classes
        self._number_of_classes = 10
        self.weight_matrix = None
        self.bias = None
        self.metric_calculator = metric_calculator

    def _softmax(self, raw_model_data):
        """
        # Softmax converts raw model scores into probabilities

        Softmax uses exponentials. Large numbers cause overflow. Subtracting the maximum value in each row prevents it.
        This does not change the result because softmax is shift-invariant.
        Example:
            [1000, 999, 995] which cause overflow
            After subtracting max (1000), we have [0, -1, -5]
            Exponentials are now safe [1, e^-1, e^-5]

        finally dvides each exponential by the sum of exponentials in the same row. This makes probabilities.        
        Example:
            exp row = [1.0, 0.37, 0.0067]
            sum     = 1.3767
            softmax = [1/1.3767, 0.37/1.3767, 0.0067/1.3767]
            All will sum to 1.
        """
        safe_raw_model_data = raw_model_data - np.max(raw_model_data, axis=1, keepdims=True)
        # Applies the exponential function to each element
        exp = np.exp(safe_raw_model_data)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _generate_one_hot_label_data(self, label_data):
        """
        Converts each label to one-hot encoded vector
        Example:
            [1, 5, 0]
            to
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        """
        number_of_labels = label_data.shape[0]
        # Empty matrix of zeros
        one_hot_data = np.zeros((number_of_labels, self._number_of_classes), dtype=np.float32)
        # np.arange(number_of_labels) selects the row and label_data selects the column which should become one.
        """
            np.arange(number_of_labels) generates 0 to number_of_labels-1 so for example if we have 5 as the fourth label,
            np.arange(number_of_labels) is 4 and refers to the fourth row of matrix and that 5 refers to the fifth item of
            that row and it becomes 1
        """
        one_hot_data[np.arange(number_of_labels), label_data] = 1.0
        return one_hot_data

    def _cross_entropy_loss(self, probabilities, one_hot_labels):
        """
        # Calculates cross-entropy loss (the loss used in softmax classification), optionally with L2 regularization.
        - Cross-entropy loss is one of the core ingredients in training classification models (like logistic regression, softmax regression, neural networks).
        - It is NOT for accuracy measurement — it is for learning.
        - It measures how “wrong” your model’s predicted probabilities are.
        - During training, your model outputs probabilities:
          Example (digit classification):
            Model says:
            Image is:
                0 → 0.01
                1 → 0.15
                2 → 0.02
                3 → 0.70   (should be correct)
                4 → 0.12
                ...
            If the true label is 3, your model should give a probability close to 1 for class 3.
            If it gives 0.70, it’s not perfect → cross-entropy becomes small but not zero.
            If it gives 0.01, it's totally wrong → cross-entropy becomes huge.
        """
        # probabilities should be batch_size x 10
        batch_size = probabilities.shape[0]
        # This ensures numerical stability because log(0) is impossible so adding eps prevents NaN or inf. 
        eps = 1e-12
        """
        This line is the core formula of cross-entropy loss for multiclass classification.
        It measures how wrong your predicted probabilities are for the true class.
        one_hot_labels tells which class is correct (1 for true class, 0 for others)
        probabilities is the model's softmax output
        log(probabilities) penalizes low probabilities
        Multiplying them picks only the probability of the correct class
        The sum averages over the whole batch
        The negative sign ensures small loss = good, big loss = bad
        Example:
            True label = 3
            One-hot:
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            Model predicts (softmax probs):
                [0.01, 0.04, 0.02, 0.60, 0.07, 0.01, 0.09, 0.04, 0.05, 0.07]
            Multiply them:
                one_hot * probs
                [0,0,0, 0.60, 0,0,0,0,0,0] <- only correct class remains
            Then:
                log(0.60) = -0.5108
                Negative: +0.5108 → loss
        """
        loss = -np.sum(one_hot_labels * np.log(probabilities + eps)) / batch_size
        """
        Why do we add this? (Intuition)
        → To prevent overfitting
            If weights become too large, model becomes too confident, memorizes training data, fails on test data.
        → L2 pushes weights to be smaller
            It works like gravity pulling weights toward zero (but not exactly zero).
            A model with smaller weights tends to be smoother and more general.
        This is the standard L2 regularization formula: (λ / 2m) × sum(W²)

        self.weight_matrix is the model’s weights (W)
        self.lambda_ is the regularization strength
            if λ = 0 → no regularization
            if λ > 0 → add penalty
        So this block runs only if L2 regularization is enabled.

        - self.weight_matrix * self.weight_matrix  squares all weights. The L2 penalty is the sum of squared weights.
        - Adds all squared weights
        - self.lambda_ / (2 * batch_size)scales the penalty:
            lambda_ controls how strong the penalty is
            dividing by 2 * batch_size keeps gradient behavior correct
        - The final added value is result of (λ / 2m) × sum(W²)
        """
        if (self.weight_matrix is not None) and (self.lambda_ > 0.0):
            loss += (self.lambda_ / (2 * batch_size)) * np.sum(self.weight_matrix * self.weight_matrix)
        return loss

    def _normalize_image_data(self, data):
        """
        Reshaping every 28×28 image to a 784-length vector then normalizing data by dividing by 255.
        (each pixel can be between 0 to 255) (-1 means all number of items in dataset)
        converting each image from
        [
        [122, 102, ..., 250],
        [122, 102, ..., 250],
        .
        .
        .
        [102, 202, ..., 255],
        ]
        which is 28x18 2D vector to
        [0.4, 0.8, ..., 1]
        784 length 1D vector.
        """
        return data.reshape(-1, 28*28).astype(np.float32) / 255.0

    def prepare_data(self):
        self._mnist_train_image = self._normalize_image_data(self._mnist_train_image)
        self._mnist_test_image = self._normalize_image_data(self._mnist_test_image)

        self._mnist_train_label_one_hot_data = self._generate_one_hot_label_data(self._mnist_train_label)
        self._mnist_test_label_one_hot_data = self._generate_one_hot_label_data(self._mnist_test_label)

    def train_model(self):
        # Number of features per image - each image is an one dimentional vector containing 784 items, so it's 784.
        number_of_features = self._mnist_train_image.shape[1]

        """
        Weight matrix of Logistic Regression contains 10 rows (for 10 numbers) and 784 columns (for 784 pixels of each number image)
        each of these rows shows weights for a digit
        it's something like:
            [
            [784 items are here like 0.0032, -0.0004],
            [784 items are here like 0.0020, 0.0007],
            .
            .
            .
            [784 items are here like 0.0011, -0.0051]
            ]
        This part of code creates a matrix filled with small random numbers. If all weights begin at zero, the model can't learn correctly (all gradients will be identical).
        Random small numbers prevent symmetry.
        """
        self.weight_matrix = np.random.normal(0, 0.01, size=(number_of_features, self._number_of_classes)).astype(np.float32)
        
        """
        Generting a one dimentional vector as bias. each item of vector shows bias of a number. 
        """
        self.bias = np.zeros((1, self._number_of_classes), dtype=np.float32)


        # Training
        # It should be 60,000 images
        number_of_training_samples = self._mnist_train_image.shape[0]
        steps_per_epoch = number_of_training_samples // batch_size

        for epoch in range(self.epochs):
            # shuffle - Shuffling avoids learning in fixed order and improves training quality.
            # perm is a random order of indices
            perm = np.random.permutation(number_of_training_samples)
            shuffled_train_images = self._mnist_train_image[perm]
            shuffled_one_hot_train_labels = self._mnist_train_label_one_hot_data[perm]
            
            # We also need the raw labels for accuracy calc (not one-hot)
            shuffled_train_labels = self._mnist_train_label[perm]

            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                batch_images = shuffled_train_images[start:end]
                batch_one_hot_labels = shuffled_one_hot_train_labels[start:end]
                number_of_batch_items = batch_images.shape[0]

                # Part 4 in initial description in this class.
                # logits should be batch_size x 10
                logits = batch_images.dot(self.weight_matrix) + self.bias
                # Part 5 in initial description in this class.
                probabilities = self._softmax(logits)

                loss = self._cross_entropy_loss(probabilities, batch_one_hot_labels)
                epoch_loss += loss

                # gradients
                # The gradient of the loss w.r.t logits is crucial because it tells the model how to adjust its weights and bias during training.
                dlogits = (probabilities - batch_one_hot_labels) / number_of_batch_items
                dW = batch_images.T.dot(dlogits)
                db = np.sum(dlogits, axis=0, keepdims=True)

                # add L2 gradient
                if self.lambda_ > 0:
                    dW += (self.lambda_ / number_of_batch_items) * self.weight_matrix

                # parameter update
                self.weight_matrix -= self.learning_rate * dW
                self.bias -= self.learning_rate * db

            epoch_loss /= steps_per_epoch
            
            # evaluate train accuracy on a subset (cheap) or on whole set if desired
            logits_train_full = self._mnist_train_image.dot(self.weight_matrix) + self.bias
            y_pred_train = np.argmax(self._softmax(logits_train_full), axis=1)
            
            # Use the new Class for Accuracy
            train_acc = self.metric_calculator.accuracy(self._mnist_train_label, y_pred_train)
            
            print(f"Epoch {epoch+1}/{self.epochs}  Loss={epoch_loss:.4f}  TrainAcc={train_acc:.4f}")

        # ---- Evaluation on test ----
        logits_test = self._mnist_test_image.dot(self.weight_matrix) + self.bias
        probs_test = self._softmax(logits_test)
        y_pred_test = np.argmax(probs_test, axis=1)

        # Use the new Class for Test Accuracy
        test_acc = self.metric_calculator.accuracy(self._mnist_test_label, y_pred_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Use the new Class for Full Report
        self.metric_calculator.print_classification_report(self._mnist_test_label, y_pred_test, self._number_of_classes)


if __name__ == "__main__":
    dm = MnistDataManager()
    dm.save_mnist()
    mnist_train_image, mnist_train_label, mnist_test_image, mnist_test_label = dm.load_data()
    learning_rate = 0.1
    epochs = 10
    batch_size = 256
    lambda_ = 1e-4
    lr = MultiClassLogisticRegression(mnist_train_label=mnist_train_label, mnist_test_label=mnist_test_label, mnist_train_image=mnist_train_image, mnist_test_image=mnist_test_image, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, metric_calculator=ClassificationMetrics, lambda_=lambda_)
    lr.prepare_data()
    lr.train_model()