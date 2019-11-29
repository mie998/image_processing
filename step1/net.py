import math
import numpy as np
from numpy.random import normal
from mnist import MNIST


class NeuralNet(object):

    def __init__(self, idx, mid_node_num):
        image_size = 28
        class_size = 10
        n1 = image_size ** 2
        n2 = mid_node_num
        n3 = class_size

        np.random.seed(10)
        w1 = normal(loc=0, scale=1 / (math.sqrt(n1)), size=(n2, n1))
        w2 = normal(loc=0, scale=1 / (math.sqrt(n2)), size=(n3, n2))
        b1 = normal(loc=0, scale=1 / (math.sqrt(n1)), size=(n2, 1))
        b2 = normal(loc=0, scale=1 / (math.sqrt(n2)), size=(n3, 1))

        self.image_idx = idx
        self.image_size = image_size
        self.mid_node_num = mid_node_num
        self.class_size = class_size
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def pre_process(self):
        mndata = MNIST('../data/')

        X, Y = mndata.load_testing()
        X = np.array(X)
        X = X[self.image_idx].reshape(-1, )

        return X

    def mid_process(self):
        X = self.pre_process()
        y = self.sigmoid(X)
        return y

    def post_process(self):
        y1 = self.mid_process()
        class_size = self.class_size

        a = []
        for i in range(class_size):
            a.append(np.dot(y1, self.w2[i]) + self.b2[i])

        a = np.array(a)
        return a

    def output(self):
        a = self.post_process()
        alpha = max(a)

        y2 = self.soft_max(map(lambda x: x - alpha, a))
        return np.argmax(y2)

    def sigmoid(self, vec):
        y = []
        size = self.mid_node_num
        for i in range(size):
            y.append(1 / (1 + math.exp(-(np.dot(self.w1[i], vec) + self.b1[i]))))

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
    idx = int(input('select index of image in range 1 ~ 9999\n'))
    assert 0 <= idx <= 9999, 'error: input integer from 1 to 9999!'

    mid_node_num = 10

    NN = NeuralNet(idx, mid_node_num)
    answer = NN.output()

    print(answer)


if __name__ == '__main__':
    main()
