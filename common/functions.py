import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:  # reshape for access with shape[0]
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    sigma = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + sigma)) / batch_size


def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)


def to_one_hot_vector(num, len):
    v = np.zeros(len)
    v[num] += 1
    return v
