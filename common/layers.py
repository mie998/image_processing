import os
import sys

sys.path.append(os.pardir)
from common.functions import *


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class Dropout:
    def __init__(self, drop_rate=0.3, is_test=False):
        self.drop_rate = drop_rate
        self.is_test = is_test
        self.banish = None

    def forward(self, x):
        self.banish = np.random.choice(int(x.shape[0]), int(x.shape[0] * self.drop_rate))
        out = x.copy()
        if self.is_test:
            out[self.banish] = 0
        else:
            out[self.banish] *= 1 - self.drop_rate

        return out

    def backward(self, dout):
        dout[self.banish] = 0
        dx = dout

        return dx


class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # output of softmax function
        self.t = None  # test data with one-hot vector

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
