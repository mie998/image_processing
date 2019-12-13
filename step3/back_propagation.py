import numpy as np
from collections import OrderedDict
import os
import sys

sys.path.append(os.pardir)
from common.layers import *
from common.utils import *


class ThreeLayerNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {
            'w1': random_array_generator_normal(input_size, hidden_size)[0],
            'b1': random_array_generator_normal(input_size, hidden_size)[1],
            'w2': random_array_generator_normal(hidden_size, output_size)[0],
            'b2': random_array_generator_normal(hidden_size, output_size)[1]}
        self.layers = OrderedDict()
        self.layers['affine1'] = Affine(w=self.params['w1'], b=self.params['b1'])
        self.layers['sigmoid'] = Sigmoid()
        self.layers['affine2'] = Affine(w=self.params['w2'], b=self.params['b2'])
        self.lastLayer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    # def loss(self, x, t):

    # def accuracy(self, x, t):

    # def gradient(self, x, t):


def main():
    np.random.seed(10)

    iteration = 100
    batch_size = 100
    input_size = 784
    hidden_size = 100
    output_size = 10
    learning_rate = 0.01

    train_losses = []

    if os.path.isfile('train_data.npz') and os.path.isfile('test_data.npz'):
        train_x, train_y = \
            np.load('train_data.npz')['x'], np.load('train_data.npz')['y']
        test_x, test_y = \
            np.load('test_data.npz')['x'], np.load('test_data.npz')['y']
    else:
        train_x, train_y, test_x, test_y = create_cache()

    print(train_x)

    for i in range(iteration):
        batch_idxes = np.random.choice(60000, batch_size)
        train_x_batch = train_x[batch_size]
        train_y_batch = train_y[batch_size]


if __name__ == '__main__':
    main()
