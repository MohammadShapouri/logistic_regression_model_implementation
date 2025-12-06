import numpy as np
import gzip
import pickle
import os

class MnistDataManager:
    def __init__(self, mnist_path="../dataset"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.mnist_path = os.path.join(script_dir, mnist_path)
        self.filename = [
            ["training_images","train-images-idx3-ubyte.gz"],
            ["test_images","t10k-images-idx3-ubyte.gz"],
            ["training_labels","train-labels-idx1-ubyte.gz"],
            ["test_labels","t10k-labels-idx1-ubyte.gz"]
        ]

    def save_mnist(self):
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(self.mnist_path + "/" + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        for name in self.filename[-2:]:
            with gzip.open(self.mnist_path + "/" + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.mnist_path + "/" + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist,f)
        print("Saving MNIST data Finished.")

    def load_data(self):
        with open(self.mnist_path + "/" + "mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
