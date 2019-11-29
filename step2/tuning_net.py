import math
import numpy as np
from numpy.random import normal
from mnist import MNIST


def cross_entropy_error(y, t):
    delta = 1e-8
    return -np.sum(t * np.log(y + delta))


def to_one_hot_vector(num, len):
    v = np.zeros(len)
    v[num] += 1
    return v


def random_array_generator_normal(size):
    n = size[0]
    m = size[1]
    w = normal(loc=0, scale=1 / (math.sqrt(n)), size=(m, n))
    b = normal(loc=0, scale=1 / (math.sqrt(n)), size=(m, 1))

    return w, b


class NeuralNet(object):

    def __init__(self, idx, mid_node_num):
        image_size = 28
        class_size = 10

        self.image_idx = idx
        self.image_size = image_size
        self.mid_node_num = mid_node_num
        self.class_size = class_size
        self.Y = []

    def pre_process(self):
        mndata = MNIST('../data/')
        idx = self.image_idx
        size = self.class_size

        X, Y = mndata.load_training()
        X = np.array(X)
        X = X[idx].reshape(-1, )

        self.Y = to_one_hot_vector(Y[idx], size)

        return X

    def mid_process(self):
        X = self.pre_process()
        y = self.sigmoid(X)
        return y

    def post_process(self):
        y1 = self.mid_process()
        class_size = self.class_size

        w, b = random_array_generator_normal((len(y1), class_size))
        a = []
        for i in range(class_size):
            a.append(np.dot(y1, w[i]) + b[i])

        a = np.array(a)
        return a

    def output(self):
        a = self.post_process()
        alpha = max(a)

        y2 = self.soft_max(map(lambda x: x - alpha, a))
        cls = self.Y

        return cross_entropy_error(y2, cls)

    def sigmoid(self, vec):
        y = []
        n = self.image_size
        m = self.mid_node_num
        w, b = random_array_generator_normal((n**2, m))
        for i in range(m):
            y.append(1 / (1 + math.exp(-(np.dot(w[i], vec) + b[i]))))

        y = np.array(y)
        return y

    def soft_max(self, vec):
        y = []
        vec = list(map(lambda x: np.exp(x), vec))
        exp_sum = sum(vec)
        for i in range(len(vec)):
            y.append(vec[i] / exp_sum)

        y = np.array(y)
        return y


def main():
    np.random.seed(10)
    number = int(input('how many images do you want to use for learning?\n'))
    assert 0 <= number <= 60000, 'error: input integer from 1 to 60000!'

    idxes = np.random.choice(60000, number)
    mid_node_num = 10

    entropy = 0
    for idx in idxes:
        NN = NeuralNet(idx, mid_node_num)
        entropy += NN.output()

    print('mean of cross-entropy: {}'.format(entropy / float(number)))


if __name__ == '__main__':
    main()
