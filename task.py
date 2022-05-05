import math
import pandas as pd
import numpy as np
from tqdm import tqdm


def scale(X_train, X_test):
    for column in X_train.columns:
        X_train[column] = X_train[column] / max(X_train[column])
        X_test[column] = X_test[column] / max(X_test[column])

    return X_train, X_test


def xavier(n_in, n_out):
    low = -math.sqrt(6) / math.sqrt(n_in + n_out)
    high = math.sqrt(6) / math.sqrt(n_in + n_out)

    return np.random.uniform(low, high, (n_in, n_out))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mse_der(y_pred, y_true):
    return 2 * (y_pred - y_true)


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def one_hot(data):
    y = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y[rows, data] = 1
    return y


def epoch_execute(X_train, y_train, estimator, batch=100, alpha=0.5):
    for i in range(0, X_train.shape[0], batch):
        _ = estimator.forward(X_train[i:i+batch, :])
        estimator.backprop(X_train[i:i+batch, :], y_train[i:i+batch, :], alpha)

    loss = mse(estimator.forward(X_train), y_train)
    return loss


def calculate_accuracy(X_test, y_test, estimator):
    predictions = estimator.forward(X_test).argmax(1)

    return (predictions == y_test).mean()


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def forward(self, X):
        self.output = sigmoid(np.dot(X, self.weights) + self.biases)

        return self.output

    def backprop(self, X, y, alpha):
        dw = np.dot(X.T, (2 * (self.output - y) * sigmoid_der(np.dot(X, self.weights) + self.biases))) / X.shape[0]
        db = np.sum((2 * (self.output - y) * sigmoid_der(np.dot(X, self.weights) + self.biases)), axis=0,
                    keepdims=True) / X.shape[0]

        self.weights -= alpha * dw
        self.biases -= alpha * db

        return mse(self.forward(X), y)


class TwoLayerNeural:
    def __init__(self, n_features, n_classes):
        self.w_1 = xavier(n_features, 64)
        self.b_1 = xavier(1, 64)
        self.w_1 = xavier(n_features, 64)
        self.b_1 = xavier(1, 64)
        self.w_2 = xavier(64, n_classes)
        self.b_2 = xavier(1, n_classes)

    def forward(self, X):
        self.layer_1 = sigmoid(np.dot(X, self.w_1) + self.b_1)
        self.output = sigmoid(np.dot(self.layer_1, self.w_2) + self.b_2)

        return self.output

    def backprop(self, X, y, alpha):
        dw_2 = np.dot(self.layer_1.T,
                      (2 * (self.output - y) * sigmoid_der(np.dot(self.layer_1, self.w_2) + self.b_2))) / \
                X.shape[0]
        db_2 = np.sum((2 * (self.output - y) * sigmoid_der(np.dot(self.layer_1, self.w_2) + self.b_2)), axis=0,
                      keepdims=True) / X.shape[0]

        dw_1 = np.dot(X.T, np.dot((2 * (self.output - y) * sigmoid_der(np.dot(self.layer_1, self.w_2) + self.b_2)),
                                  self.w_2.T) * sigmoid_der(np.dot(X, self.w_1) + self.b_1)) / X.shape[0]

        db_1 = np.sum(np.dot((2 * (self.output - y) * sigmoid_der(np.dot(self.layer_1, self.w_2) + self.b_2)),
                             self.w_2.T) * sigmoid_der(np.dot(X, self.w_1) + self.b_1), axis=0, keepdims=True) / \
            X.shape[0]

        self.w_2 -= alpha * dw_2
        self.b_2 -= alpha * db_2
        self.w_1 -= alpha * dw_1
        self.b_1 -= alpha * db_1

        return mse(self.forward(X), y)


train_set = pd.read_csv("scaled-fashion-mnist_train.csv")
test_set = pd.read_csv("scaled-fashion-mnist_test.csv")

y_train = one_hot(train_set["label"].values)
X_train = train_set.drop(["label"], axis=1).values

y_test = one_hot(test_set["label"].values)
X_test = test_set.drop(["label"], axis=1).values

# X_train, X_test = scale(X_train, X_test)

model = TwoLayerNeural(784, 10)
accuracy_list = list()

for i in tqdm(range(20)):
    _ = epoch_execute(X_train, y_train, model, 100, 0.5)
    accuracy = calculate_accuracy(X_test, y_test.argmax(1), model)
    accuracy_list.append(accuracy)

print(accuracy_list)
