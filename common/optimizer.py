import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= grads[key] * self.lr


class Momentum:
    def __init__(self, alpha=0.9, lr=0.01):
        self.alpha = alpha
        self.lr = lr
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            velocity = {}
            for key, val in params.keys():
                velocity[key] = np.zeros_like(val)
            self.velocity = velocity

        for key in params.keys():
            self.velocity[key] = self.alpha * self.velocity[key] - grads[key] * self.lr
            params[key] += self.velocity[key]


class AdaGrad:
    def __init__(self, lr=0.001, delta=1e-8):
        self.lr = lr
        self.delta = delta
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            h = {}
            for key, val in params.keys():
                h[key] = np.zeros_like(val)
                h[key] += self.delta
            self.h = h

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.delta)


class RMSProp:
    def __init__(self, lr=0.001, law=0.9, delta=1e-8):
        self.lr = lr
        self.law = law
        self.delta = delta
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            h = {}
            for key, val in params.keys():
                h[key] = np.zeros_like(val)
            self.h = h

        for key in params.keys():
            self.h[key] = self.law * self.h[key] + (1 - self.law) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.delta)


class AdaDelta:
    def __init__(self, law=0.95, delta=1e-6):
        self.law = law
        self.delta = delta
        self.h = None
        self.s = None
        self.dw = None

    def update(self, params, grads):
        if self.h is None or self.s is None or self.dw is None:
            h = {}
            s = {}
            dw = {}
            for key, val in params.keys():
                h[key] = np.zeros_like(val)
                s[key] = np.zeros_like(val)
                dw[key] = np.zeros_like(val)
            self.h = h
            self.s = s
            self.dw = dw

        for key in params.keys():
            self.h[key] = self.law * self.h[key] + (1 - self.law) * grads[key] * grads[key]
            self.dw[key] = - np.sqrt(self.s[key] + self.delta) / np.sqrt(self.h[key] + self.delta) * grads[key]
            self.s[key] = self.law * self.s[key] + (1 - self.law) * grads[key] * grads[key]
            params[key] += self.dw[key]


class Adam:
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, delta=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta
        self.t = 0
        self.m = None
        self.v = None
        self.m_ = None
        self.v_ = None

    def update(self, params, grads):
        if self.m is None or self.v is None or self.m_ is None or self.v_ is None:
            m = {}
            v = {}
            m_ = {}
            v_ = {}
            for key, val in params.keys():
                m[key] = np.zeros_like(val)
                v[key] = np.zeros_like(val)
                m_[key] = np.zeros_like(val)
                v_[key] = np.zeros_like(val)
            self.m = m
            self.v = v
            self.m_ = m_
            self.v_ = v_

        self.t += 1
        for key in params.key():
            self.m[key] = self.beta_1 * m + (1 - self.beta_1) * grads[key]
            self.v[key] = self.beta_2 * v + (1 - self.beta_2) * grads[key] * grads[key]
            self.m_[key] = self.m[key] / (1 - self.beta_1 ** self.t)
            self.v_[key] = self.v[key] / (1 - self.beta_2 ** self.t)
            params[key] -= self.alpha * self.m_[key] / (np.sqrt(self.v_[key]) + self.delta)
