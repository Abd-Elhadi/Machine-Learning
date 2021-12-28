import pandas as pd
import numpy as np


def gradient_descent(x, y, w, alpha, lamda, iterations):
    m = len(x)  # size of the training set

    for iteration in range(iterations):
        fx = np.dot(x, w)
        for i in range(m):
            # No mis-classification
            if y[i] * fx[i] >= 1:
                w = w - (alpha * (2 * lamda) * w)
            # Mis-classification
            else:
                w = w + (alpha * ((y[i] * x[i]) - (2 * lamda * w)))
    return w


def predict(x, w):
    result = np.dot(x, w)
    return np.sign(result)


def accuracy(y_test, y_result):
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == y_result[i]:
            correct += 1
    ratio = correct / len(y_test)
    return ratio * 100


if __name__ == '__main__':

    input_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs']
    output_fields = ['target']
    url = "heart.csv"
    dataset = pd.read_csv(url)

    x = dataset[input_fields]
    y = dataset[output_fields]

    # Feature Scaling, Mean Normalization
    x = (x - x.mean()) / x.std()

    y = y.iloc[:, 0]
    y = y.apply(lambda v: -1 if v < 1 else 1)

    x = x.values
    y = y.values

    # split dataset into training and testing data
    size = round(len(dataset) * 0.8)
    x_train = x[:size]
    x_test = x[size:]

    y_train = y[:size]
    y_test = y[size:]
    w = np.array([0] * len(x_train[0]))  # [0 0 0 0 0]

    alpha = 0.0001
    iterations = 400
    lamda = 0.01

    w = gradient_descent(x_train, y_train, w, alpha, lamda, iterations)

    print("\nOptimized Weight : ")
    print(w)

    y_pred = predict(x_test, w)

    print("\nAccuracy: ", accuracy(y_test, y_pred), "%")
