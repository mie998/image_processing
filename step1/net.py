import math
import numpy as np
from numpy.random import normal
from mnist import MNIST


def random_array_generator_normal(size):
    n = size[0]
    m = size[1]
    w = normal(loc=0, scale=1 / (math.sqrt(n)), size=(m, n))
    b = normal(loc=0, scale=1 / (math.sqrt(n)), size=(m, 1))

    return w, b


def sigmoid(image_size, mid_node_num, vec):
    n = image_size
    m = mid_node_num
    w, b = random_array_generator_normal((n ** 2, m))

    y = [1 / (1 + math.exp(-(np.dot(w[i], vec) + b[i]))) for i in range(m)]
    y = np.array(y)

    return y


def soft_max(vec):
    vec = list(map(lambda x: np.exp(x), vec))
    exp_sum = sum(vec)

    y = [vec[i] / exp_sum for i in range(len(vec))]
    y = np.array(y)

    return y


class NeuralNet(object):

    def __init__(self, idx, mid_node_num):
        image_size = 28
        class_size = 10

        self.image_idx = idx
        self.image_size = image_size
        self.mid_node_num = mid_node_num
        self.class_size = class_size

    def pre_process(self):
        mndata = MNIST('../data/')

        X, Y = mndata.load_testing()
        X = np.array(X)
        X = X[self.image_idx].reshape(-1, )

        return X

    def mid_process(self):
        X = self.pre_process()
        y = sigmoid(self.image_size, self.mid_node_num, X)
        return y

    def post_process(self):
        y1 = self.mid_process()
        class_size = self.class_size

        w, b = random_array_generator_normal((len(y1), class_size))
        a = [np.dot(y1, w[i]) + b[i] for i in range(class_size)]
        a = np.array(a)

        return a

    def output(self):
        a = self.post_process()
        alpha = max(a)

        y2 = soft_max(map(lambda x: x - alpha, a))
        return np.argmax(y2)


def main():
    np.seed(10)
    idx = int(input('select index of image in range 1 ~ 9999\n'))
    assert 0 <= idx <= 9999, 'error: input integer from 1 to 9999!'

    mid_node_num = 50

    NN = NeuralNet(idx, mid_node_num)
    answer = NN.output()

    print(answer)


if __name__ == '__main__':
    main()
