import os
import sys

sys.path.append(os.pardir)
from common.functions import *
from common.utils import *


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
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        if x.ndim > 2:
            self.x = x.reshape(x.shape[0], -1)
        else:
            self.x = x
        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(self.input_shape)

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


class BatchNormalization:
    def __init__(self, beta=0.0, gamma=1.0, delta=1e-7, is_test=False):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.is_test = is_test
        self.batch_size = None
        self.mean = None
        self.variance = None
        self.x = None
        self.xvar = None
        self.dbeta = None
        self.dgamma = None

    def forward(self, x):
        self.batch_size = x.shape[0]
        node_size = x.shape[1]
        self.x = x
        mean_node = sum(x) / node_size
        mean = sum(mean_node) / self.batch_size
        variance_node = sum((x - mean) ** 2) / node_size
        variance = sum(variance_node) / self.batch_size
        self.mean = mean
        self.variance = variance
        self.xvar = (x - mean) / np.sqrt(variance + self.delta)
        if not self.is_test:
            out = self.gamma * self.xvar + self.beta
        else:
            out = self.gamma / np.sqrt(variance + self.delta) * x + \
                  self.beta - self.gamma * mean / (variance + self.delta)
        return out

    def backward(self, dout):
        d_xvar = self.gamma * dout
        d_variance = -1 / 2 * (self.variance + self.delta) ** (-3 / 2) * \
                     np.sum((self.x - self.mean) * d_xvar)
        d_mean = -1 / np.sqrt(self.variance + self.gamma) * d_xvar + \
                 -2 * d_variance * np.sum(self.x - self.mean) / self.batch_size
        dx = d_xvar * 1 / np.sqrt(self.variance + self.gamma) + \
             d_variance * 2 * (self.x - self.mean) / self.batch_size + \
             d_mean / self.batch_size
        self.dgamma = np.sum(dout * self.xvar)
        self.dbeta = dout.sum(axis=0)

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


class Convolution:
    def __init__(self, w, b, stride=1, padding=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding
        self.x = None
        self.col = None
        self.col_w = None
        self.dw = None
        self.db = None

    def forward(self, x):
        N, C, H, W = x.shape
        FN, C, FH, FW = self.w.shape
        out_h = (H + 2 * self.padding - FH) // self.stride + 1
        out_w = (W + 2 * self.padding - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.padding)
        col_w = self.w.reshape(-1, FN)
        self.x = x
        self.col = col
        self.col_w = col_w

        out = np.dot(col, col_w) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        dw_col = np.dot(self.col.T, dout)
        db_col = np.sum(dout, axis=0)
        dx_col = np.dot(dout, self.col_w.T)
        self.dw = dw_col.reshape(FN, C, FH, FW)
        self.db = db_col
        dx = col2im(dx_col, self.x.shape, FH, FW, self.stride, self.padding)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding
        self.x = None
        self.x_col = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.padding - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        arg_max = np.argmax(col, axis=1)
        col_out = np.max(col, axis=1)
        out = col_out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.x_col = col_out
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        col_dout = dout.reshape(-1, self.pool_h * self.pool_w)
        col_dx = np.zeros((dout.size, self.pool_h * self.pool_w))
        col_dx[self.arg_max] = col_dout[self.arg_max]

        dx = col2im(col_dx, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)

        return dx
