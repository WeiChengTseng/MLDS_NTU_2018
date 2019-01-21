import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
try:
    plt.style.use('gadfly')
except:
    pass


class ShallowNet():
    def __init__(self, in_dim=1, out_dim=1, model_type='regression'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
        self._hidden_layer = tf.layers.dense(inputs=self.x, units=64, activation=tf.nn.relu)
        self._out_layer = tf.layers.dense(inputs=self._hidden_layer, units=out_dim)
        if model_type=='regression':
            self._loss = tf.losses.mean_squared_error(self._out_layer, self.y)
        else:
            self._loss = tf.losses.softmax_cross_entropy(self.y, self._out_layer)
        self._opt = tf.train.GradientDescentOptimizer(1e-2)
        self._train_step = self._opt.minimize(self._loss)
        return

    def train(self, x, y, batch_size=5000, epoch=2000):
        bs = batch_size if batch_size < len(x) else 50
        num_batch = len(x) // bs
        his = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm.tqdm(range(epoch)):
                for j in range(num_batch):
                    fd = {self.x: x[j*bs: (j+1)*bs], self.y: y[j*bs: (j+1)*bs]}
                    sess.run(self._train_step, feed_dict=fd)
                his.append(sess.run(self._loss, feed_dict=fd))
                choice = np.random.choice(len(x), len(x), replace=False)
                x, y = x[choice], y[choice]
            pred = sess.run(self._out_layer, feed_dict={self.x: x, self.y: y})
        return his, pred

class DeepNet():
    def __init__(self, in_dim=1, out_dim=1, model_type='regression'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
        self._hidden_layer = tf.layers.dense(self.x, 7, activation=tf.nn.relu)
        self._hidden_layer2 = tf.layers.dense(self._hidden_layer, 7, activation=tf.nn.relu)
        self._out_layer = tf.layers.dense(self._hidden_layer, out_dim)
        if model_type=='regression':
            self._loss = tf.losses.mean_squared_error(self._out_layer, self.y)
        else:
            self._loss = tf.losses.softmax_cross_entropy(self.y, self._out_layer)
        self._opt = tf.train.GradientDescentOptimizer(1e-2)
        self._train_step = self._opt.minimize(self._loss)
        return
    
    def train(self, x, y, batch_size=5000, epoch=2000):
        bs = batch_size if batch_size < len(x) else 50
        num_batch = len(x) // bs
        his = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm.tqdm(range(epoch)):
                for j in range(num_batch):
                    fd = {self.x: x[j*bs: (j+1)*bs], self.y: y[j*bs: (j+1)*bs]}
                    sess.run(self._train_step, feed_dict=fd)
                his.append(sess.run(self._loss, feed_dict=fd))
                choice = np.random.choice(len(x), len(x), replace=False)
                x, y = x[choice], y[choice]
            pred = sess.run(self._out_layer, feed_dict={self.x: x, self.y: y})
        return his, pred

def plot(his1, his2, name='mnist'):
    plt.plot(np.arange(len(his1)), his1, label='shallow', alpha=0.4)
    plt.plot(np.arange(len(his2)), his2, label='deep', alpha=0.4)
    plt.legend()
    
    plt.ylim(0, 10)
    # plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('{} of mnist with different layers'.format(name))
    
    plt.savefig('hw1-1/{}.png'.format(name), dpi=600)
    plt.close()
    return

def simulate(pred1, pred2):
    x_sinc_ = np.linspace(-10, 10, 500) * 5 * 3.1415926
    y_sinc_ = np.sinc(x_sinc_)
    x_ = np.linspace(-10, 10, 50) * 5 * 3.1415926
    plt.plot(x_, pred1, label='shallow')
    plt.plot(x_, pred2, label='deep')
    plt.plot(x_sinc_, y_sinc_, label='sinc(5x*pi)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('sinc(5x*pi)')
    plt.savefig('hw1-1/{}.png'.format('result'), dpi=600)
    plt.close()
    return

if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (12,6.75)
    x_sinc = np.linspace(-10, 10, 250) * 5 * 3.1415926
    y_sinc = np.sinc(x_sinc)
    x_sinc, y_sinc = x_sinc[:, np.newaxis], y_sinc[:, np.newaxis]
    net1 = ShallowNet(out_dim=1)
    his1, pred1 = net1.train(x_sinc, y_sinc)
    net2 = DeepNet(out_dim=1)
    his2, pred2 = net2.train(x_sinc, y_sinc)
    plot(his1, his2, 'sinc(5x*pi)')
    simulate(pred1, pred2)
    del (net1, net2)
    
    (x_mnist, y_), (_, __) = tf.keras.datasets.mnist.load_data()
    x_mnist, y_mnist = x_mnist.reshape([x_mnist.shape[0], -1]), np.zeros((len(x_mnist), 10))
    y_mnist[np.arange(60000), y_] = 1
    net1 = ShallowNet(in_dim=28*28, out_dim=10, model_type='classification')
    his1, _ = net1.train(x_mnist, y_mnist)
    net2 = DeepNet(in_dim=28*28, out_dim=10, model_type='classification')
    his2, _ = net2.train(x_mnist, y_mnist)
    plot(his1, his2)