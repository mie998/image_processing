import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
from numpy.random import normal
import os
import sys

sys.path.append(os.pardir)
from step3.back_propagation import *


def main():
    idx = int(input('select index of image in range 1 ~ 9999\n'))
    assert 0 <= idx <= 9999, 'error: input integer from 1 to 9999!'

    _, _, test_x, test_y = read_MNIST()
    test_x_img = test_x[idx]
    test_y_ans = test_y[idx]

    w1, b1, w2, b2 = read_parameter('../data/learned_parameter.npz')
    input_size = 784
    hidden_size = 50
    output_size = 10

    NN = ThreeLayerNeuralNet(input_size, hidden_size, output_size)
    NN.params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    NN.layers['affine1'].w = w1
    NN.layers['affine1'].b = b1
    NN.layers['affine2'].w = w2
    NN.layers['affine2'].b = b2
    x = NN.predict(test_x_img)
    guess = np.argmax(x)

    print('I guess the image shows... {}'.format(guess))
    print('real number was {}'.format(test_y_ans))

    test_x_img = test_x_img.reshape(28, 28)
    plt.imshow(test_x_img, cmap=cm.gray)
    plt.show()


if __name__ == '__main__':
    main()
