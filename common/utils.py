from mnist import MNIST
import math
import numpy as np
from numpy.random import normal


def random_array_generator_normal(din_size, dout_size):
    # usage: np.dot(x, w) + b
    n, m = din_size, dout_size
    w = normal(loc=0, scale=1 / (math.sqrt(n)), size=(n, m))
    b = normal(loc=0, scale=1 / (math.sqrt(n)), size=(m, 1))

    return w, b


def create_cache():
    mndata = MNIST('../data/')
    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()
    train_x, train_y, test_x, test_y = \
        np.array(train_x), np.array(train_x), np.array(train_x), np.array(train_x)
    np.savez('train_data.npz', x=train_x, y=train_y)
    np.savez('test_data.npz', x=test_x, y=test_y)

    return train_x, train_y, test_x, test_y
