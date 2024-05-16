import pandas as pd
import numpy as np


def initialize():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)

    return w1, b1, w2, b2

def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return x > 0

def softmax(x):
    a = np.exp(x) / np.sum(np.exp(x), axis = 0, keepdims = True)
    return a

def forward(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(Y):
    onehotY = np.zeros((Y.size, Y.max() + 1))
    onehotY[np.arange(Y.size), Y] = 1
    return onehotY.T

def backward(z1, a1, z2, a2, w1, w2, x, y):
    m = y.size 
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis = 1, keepdims = True)
    dz1 = w2.T.dot(dz2) * deriv_relu(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis = 1, keepdims = True)
    
    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2

    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, alpha, iterations):
    w1, b1, w2, b2 = initialize()
    for i in range(iterations):
        z1, a1, z2, a2 = forward(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 500 == 0:
            print("Iteration:", i)
            predictions = get_predictions(a2)
            print("Accuracy:", get_accuracy(predictions, y))
    return w1, b1, w2, b2


data = pd.read_csv('mnist_test.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_, m_train = X_train.shape

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.10, 10001)
