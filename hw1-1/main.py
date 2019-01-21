import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.style.use('gadfly')
except:
    pass


class ShallowNet():
    def __init__(self, in_dim=1, out_dim=1, model_type='regression'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
        self._hidden_layer = tf.layers.dense(inputs=self.x, units=5, activation=tf.nn.relu)
        self._out_layer = tf.layers.dense(inputs=self._hidden_layer, units=out_dim)
        if model_type=='regression':
            self._loss = tf.losses.mean_squared_error(self._out_layer, self.y)
        else:
            self._loss = tf.losses.softmax_cross_entropy(self.y, self._out_layer)
        self._opt = tf.train.AdamOptimizer()
        self._train_step = self._opt.minimize(self._loss)
        return

    def train(self, x, y, batch_size=100, epoch=100):
        bs = batch_size
        his = []
        print(y.shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                fd = {self.x: x[i*bs: (i+1)*bs], self.y: y[i*bs: (i+1)*bs]}
                sess.run(self._train_step, feed_dict=fd)
                his.append(sess.run(self._loss, feed_dict=fd))
        return his

class DeepNet():
    def __init__(self, in_dim=1, out_dim=1, model_type='regression'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
        self._hidden_layer = tf.layers.dense(self.x, 5, activation=tf.nn.relu)
        self._hidden_layer2 = tf.layers.dense(self._hidden_layer, 5, activation=tf.nn.relu)
        self._out_layer = tf.layers.dense(self._hidden_layer, out_dim)
        if model_type=='regression':
            self._loss = tf.losses.mean_squared_error(self._out_layer, self.y)
        else:
            self._loss = tf.losses.softmax_cross_entropy(self.y, self._out_layer)
        self._opt = tf.train.AdamOptimizer()
        self._train_step = self._opt.minimize(self._loss)
        return
    
    def train(self, x, y, batch_size=100, epoch=1000):
        bs = batch_size
        his = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                fd = {self.x: x[i*bs: (i+1)*bs], self.y: y[i*bs: (i+1)*bs]}
                sess.run(self._train_step, feed_dict=fd)
                his.append(sess.run(self._loss, feed_dict=fd))
        return his

if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (12,6.75)
    x_sinc, y_sinc = np.linspace(-5, 5, 100), np.sinc(np.linspace(-5, 5, 100))*10
    x_sinc, y_sinc = x_sinc[:, np.newaxis], y_sinc[:, np.newaxis]
    net1 = ShallowNet(out_dim=1)
    net1.train(x_sinc, y_sinc)
    net2 = DeepNet(out_dim=1)
    net2.train(x_sinc, y_sinc)
    del (net1, net2)
    
    (x_mnist, y_), (_, __) = tf.keras.datasets.mnist.load_data()
    x_mnist, y_mnist = x_mnist.reshape([x_mnist.shape[0], -1]), np.zeros((len(x_mnist), 10))
    y_mnist[np.arange(60000), y_] = 1
    print(x_mnist.shape)
    net1 = ShallowNet(in_dim=28*28, out_dim=10, model_type='classification')
    his1 = net1.train(x_mnist, y_mnist)
    net2 = DeepNet(in_dim=28*28, out_dim=10, model_type='classification')
    his2 = net2.train(x_mnist, y_mnist)
    plt.plot(np.arange(len(his1)), his1, label='shallow')
    plt.plot(np.arange(len(his2)), his2, label='deep')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('model of mnist with different layers')
    plt.yscale('log')
    plt.savefig('hw1-1/mnist.png', dpi=600)