from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.pardir)
from common.layers import *
from common.utils import *
from common.optimizer import *


class ConvolutionalNeuralNet:
    def __init__(self, input_shape, hidden_size, output_size, conv_params):
        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        filter_stride = conv_params['filter_stride']
        filter_padding = conv_params['filter_padding']
        pool_size = conv_params['pool_size']
        pool_stride = conv_params['pool_stride']
        pool_padding = conv_params['pool_padding']
        channel_num = input_shape[0]
        conv_output_size = (input_shape[2] + 2 * filter_padding - filter_size) // filter_stride + 1
        pool_output_num = filter_num * ((conv_output_size + 2 * pool_padding - pool_size) // pool_stride + 1) ** 2
        self.params = {
            'w1': np.random.randn(filter_num, channel_num, filter_size, filter_size),
            'b1': np.random.randn(filter_num),
            'w2': random_array_generator_normal(pool_output_num, hidden_size)[0],
            'b2': random_array_generator_normal(pool_output_num, hidden_size)[1],
            'w3': random_array_generator_normal(hidden_size, output_size)[0],
            'b3': random_array_generator_normal(hidden_size, output_size)[1],
        }
        self.layers = OrderedDict()
        self.layers['convolution'] = Convolution(w=self.params['w1'], b=self.params['b1'],
                                                 stride=filter_stride, padding=filter_padding)
        self.layers['relu1'] = ReLU()
        self.layers['pooling'] = Pooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, padding=pool_padding)
        self.layers['affine1'] = Affine(w=self.params['w2'], b=self.params['b2'])
        self.layers['relu2'] = ReLU()
        self.layers['dropout'] = Dropout(drop_rate=0.3, is_test=False)
        self.layers['affine2'] = Affine(w=self.params['w3'], b=self.params['b3'])
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
        gradients['w1'] = self.layers['convolution'].dw
        gradients['b1'] = self.layers['convolution'].db
        gradients['w2'] = self.layers['affine1'].dw
        gradients['b2'] = self.layers['affine1'].db
        gradients['w3'] = self.layers['affine2'].dw
        gradients['b3'] = self.layers['affine2'].db

        return gradients


def main():
    np.random.seed(1)

    iteration = 10000
    batch_size = 100
    hidden_size = 100
    output_size = 10
    sifar_img_num = 10000
    sifar_channel_num = 3
    sifar_img_size = 32
    epoch_size = sifar_img_num / batch_size

    train_losses = []
    train_accs = []
    test_accs = []

    pickle = '../data/cifar-10-batches-py/'
    train_x, train_y = unpickle(pickle + 'data_batch_1')
    test_x, test_y = unpickle(pickle + 'test_batch')
    train_x = normalization(train_x)
    test_x = normalization(test_x)
    train_x = train_x.reshape(sifar_img_num, sifar_channel_num, sifar_img_size, sifar_img_size)
    test_x = test_x.reshape(sifar_img_num, sifar_channel_num, sifar_img_size, sifar_img_size)

    # img_size + padding*2 - filter_size が filter_stride の倍数になるようにパラメータを設定する
    conv_params = {
        'filter_num': 10,
        'filter_size': 5,
        'filter_stride': 2,
        'filter_padding': 1,
        'pool_size': 5,
        'pool_stride': 1,
        'pool_padding': 0,
    }
    CNN = ConvolutionalNeuralNet((sifar_channel_num, sifar_img_size, sifar_img_size),
                                 hidden_size, output_size, conv_params)

    for i in range(iteration):
        batch_idxes = np.random.choice(sifar_img_num, batch_size)
        train_x_batch = train_x[batch_idxes]
        train_y_batch = train_y[batch_idxes]
        train_y_batch = to_one_hot_vector_batch(train_y_batch, output_size)

        gradients = CNN.gradient(train_x_batch, train_y_batch)

        ### select optimizer for comparison
        # optimizer = SGD(lr=0.01)
        # optimizer = Momentum(alpha=0.9, lr=0.01)
        # optimizer = AdaGrad(lr=0.001, delta=1e-8)
        # optimizer = RMSProp(lr=0.001, law=0.9, delta=1e-8)
        # optimizer = AdaDelta(law=0.95, delta=1e-6)
        optimizer = Adam(alpha=0.001, beta_1=0.9, beta_2=0.999, delta=1e-8)

        optimizer.update(CNN.params, gradients)

        loss = CNN.loss(train_x_batch, train_y_batch)
        print("loss: {}".format(loss))

        if i % epoch_size == 0:
            train_acc = CNN.accuracy(train_x, train_y)
            test_acc = CNN.accuracy(test_x, test_y)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            train_losses.append(loss)
            print("----- epoch{} -----".format(i / epoch_size))
            print("train accuracy: {}%".format(train_acc * 100))
            print("test accuracy: {}%".format(test_acc * 100))

    epochs = range(len(train_accs))
    plt.plot(epochs, train_accs, 'b', label='train_acc')
    plt.plot(epochs, test_accs, 'r', label='test_acc')
    plt.title('train and test accuracy')
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
