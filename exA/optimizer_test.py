from collections import OrderedDict
import os
import sys

sys.path.append(os.pardir)
from common.layers import *
from common.utils import *
from common.optimizer import *


class ThreeLayerNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {
            'w1': random_array_generator_normal(input_size, hidden_size)[0],
            'b1': random_array_generator_normal(input_size, hidden_size)[1],
            'w2': random_array_generator_normal(hidden_size, output_size)[0],
            'b2': random_array_generator_normal(hidden_size, output_size)[1]}
        self.layers = OrderedDict()
        self.layers['affine1'] = Affine(w=self.params['w1'], b=self.params['b1'])
        self.layers['ReLU'] = ReLU()
        self.layers['dropout'] = Dropout()
        self.layers['affine2'] = Affine(w=self.params['w2'], b=self.params['b2'])
        self.lastLayer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        ans = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(ans == t) / float(x.shape[0])

        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        gradients = {}
        gradients['w1'] = self.layers['affine1'].dw
        gradients['b1'] = self.layers['affine1'].db
        gradients['w2'] = self.layers['affine2'].dw
        gradients['b2'] = self.layers['affine2'].db

        return gradients


def main():
    np.random.seed(1)

    iteration = 10000
    batch_size = 100
    input_size = 784
    hidden_size = 50
    output_size = 10
    image_size = 60000
    epoch_size = image_size / batch_size

    train_losses = []
    train_grads = []
    train_accs = []

    train_x, train_y, test_x, test_y = read_MNIST()
    train_x = normalization(train_x)
    test_x = normalization(test_x)

    NN = ThreeLayerNeuralNet(input_size, hidden_size, output_size)
    for i in range(iteration):
        batch_idxes = np.random.choice(image_size, batch_size)
        train_x_batch = train_x[batch_idxes]
        train_y_batch = train_y[batch_idxes]
        train_y_batch = to_one_hot_vector_batch(train_y_batch, output_size)

        gradients = NN.gradient(train_x_batch, train_y_batch)

        ### select optimizer for comparison
        # optimizer = SGD(lr=0.01)
        # optimizer = Momentum(alpha=0.9, lr=0.01)
        # optimizer = AdaGrad(lr=0.001, delta=1e-8)
        # optimizer = RMSProp(lr=0.001, law=0.9, delta=1e-8)
        # optimizer = AdaDelta(law=0.95, delta=1e-6)
        optimizer = Adam(alpha=0.001, beta_1=0.9, beta_2=0.999, delta=1e-8)

        optimizer.update(NN.params, gradients)

        loss = NN.loss(train_x_batch, train_y_batch)

        if i % epoch_size == 0:
            train_acc = NN.accuracy(train_x, train_y)
            test_acc = NN.accuracy(test_x, test_y)
            train_accs.append(train_acc)
            train_losses.append(loss)
            train_grads.append(gradients)
            print("----- epoch{} -----".format(i / epoch_size))
            print("loss: {}".format(loss))
            print("train accuracy: {}%".format(train_acc * 100))
            print("test accuracy: {}%".format(test_acc * 100))


if __name__ == '__main__':
    main()
