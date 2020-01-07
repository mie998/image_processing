from mnist import MNIST
import math
import os
import numpy as np
from numpy.random import normal


def random_array_generator_normal(din_size, dout_size):
    # usage: np.dot(x, w) + b
    n, m = din_size, dout_size
    w = normal(loc=0, scale=1 / (np.sqrt(n)), size=(n, m))
    b = normal(loc=0, scale=1 / (np.sqrt(n)), size=m)

    return w, b


def normalization(data):
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min)
    return data


def create_cache():
    mndata = MNIST('../data/')
    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()
    train_x, train_y, test_x, test_y = \
        np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
    np.savez('../data/train_data.npz', x=train_x, y=train_y)
    np.savez('../data/test_data.npz', x=test_x, y=test_y)

    return train_x, train_y, test_x, test_y


def read_MNIST():
    if os.path.isfile('../data/train_data.npz') and os.path.isfile('../data/test_data.npz'):
        train_x, train_y = \
            np.load('../data/train_data.npz')['x'], np.load('../data/train_data.npz')['y']
        test_x, test_y = \
            np.load('../data/test_data.npz')['x'], np.load('../data/test_data.npz')['y']
    else:
        train_x, train_y, test_x, test_y = create_cache()

    return train_x, train_y, test_x, test_y


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.array(dict[b'data'])
    X.reshape((X.shape[0], 3, 32, 32))
    Y = np.array(dict[b'labels'])
    return X, Y


def save_parameter(object):
    w1, b1, w2, b2 = object.params['w1'], object.params['b1'], object.params['w2'], object.params['b2']
    np.savez('../data/learned_parameter.npz', w1=w1, b1=b1, w2=w2, b2=b2)


def read_parameter(file):
    if os.path.isfile(file):
        w1 = np.load(file)['w1']
        b1 = np.load(file)['b1']
        w2 = np.load(file)['w2']
        b2 = np.load(file)['b2']

        return w1, b1, w2, b2
    else:
        print("can't read parameter data")


def to_one_hot_vector(num, output_size):
    v = np.zeros(output_size)
    v[num] += 1

    return v


def to_one_hot_vector_batch(vec, output_size):
    v = np.zeros((vec.size, output_size))
    for i in range(vec.size):
        v[i, vec[i]] += 1

    return v


def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for h in range(filter_h):
        h_bound = h + stride * out_h
        for w in range(filter_w):
            w_bound = w + stride * out_w
            col[:, :, h, w, :, :] = img[:, :, h:h_bound:stride, w:w_bound:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, padding=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))

    for h in range(filter_h):
        h_bound = h + stride * out_h
        for w in range(filter_w):
            w_bound = w + stride * out_w
            img[:, :, h:h_bound:stride, w:w_bound:stride] += col[:, :, h, w, :, :]

    return img[:, :, padding:(H + padding), padding:(W + padding)]
