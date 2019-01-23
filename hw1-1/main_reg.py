import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tqdm
import os
import pickle
try:
    plt.style.use('gadfly')
except:
    pass
try:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
except:
    print('no GPU available')
    pass

class ShallowNet():
    def __init__(self, in_dim=1, out_dim=1, model_type='regression'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
        self._hidden_layer = tf.layers.dense(inputs=self.x, units=190, activation=tf.nn.relu)
        self._out_layer = tf.layers.dense(inputs=self._hidden_layer, units=out_dim)
        if model_type=='regression':
            self._loss = tf.losses.mean_squared_error(self._out_layer, self.y)
        else:
            self._loss = tf.losses.softmax_cross_entropy(self.y, self._out_layer)
        self._opt = tf.train.AdamOptimizer()
        self._train_step = self._opt.minimize(self._loss)
        return

    def train(self, x, y, batch_size=5000, epoch=20000):
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
            pred = sess.run(self._out_layer, feed_dict={self.x: x})
        return his, pred

class DeepNet():
    def __init__(self, in_dim=1, out_dim=1, model_type='regression'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
        self._hidden_layer = tf.layers.dense(self.x, 10, activation=tf.nn.relu)
        self._hidden_layer2 = tf.layers.dense(self._hidden_layer, 18, activation=tf.nn.relu)
        self._hidden_layer3 = tf.layers.dense(self._hidden_layer2, 15, activation=tf.nn.relu)
        self._hidden_layer4 = tf.layers.dense(self._hidden_layer3, 4, activation=tf.nn.relu)
        self._out_layer = tf.layers.dense(self._hidden_layer4, out_dim)
        if model_type=='regression':
            self._loss = tf.losses.mean_squared_error(self._out_layer, self.y)
        else:
            self._loss = tf.losses.softmax_cross_entropy(self.y, self._out_layer)
        self._opt = tf.train.AdamOptimizer()
        self._train_step = self._opt.minimize(self._loss)
        return
    
    def train(self, x, y, batch_size=5000, epoch=20000):
        bs = batch_size if batch_size < len(x) else 50
        num_batch = len(x) // bs
        x_copy, y_copy, his = np.copy(x), np.copy(y), []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm.tqdm(range(epoch)):
                for j in range(num_batch):
                    fd = {self.x: x[j*bs: (j+1)*bs], self.y: y[j*bs: (j+1)*bs]}
                    sess.run(self._train_step, feed_dict=fd)
                his.append(sess.run(self._loss, feed_dict=fd))
                choice = np.random.choice(len(x), len(x), replace=False)
                x, y = x[choice], y[choice]
            pred = sess.run(self._out_layer, feed_dict={self.x: x_copy, self.y: y_copy})
        return his, pred

def plot(his1, his2, name='mnist', limit=10):
    pickle.dump({'shallow': his1, 'deep': his2}, open('hw1-1/his.p', 'wb'))
    plt.plot(np.arange(len(his1)), his1, label='shallow', alpha=0.7)
    plt.plot(np.arange(len(his2)), his2, label='deep', alpha=0.7)
    plt.legend()
    plt.xlim(left=10)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('{} of mnist with different layers'.format(name))
    
    plt.savefig('hw1-1/{}.png'.format(name), dpi=600)
    plt.close()
    return

def simulate(pred1, pred2):
    x_sinc_ = np.linspace(-1, 1, 400)
    y_sinc_ = np.sinc(x_sinc_ * 5)
    x_ = x_sinc_
    plt.plot(x_, pred1, label='shallow', alpha=0.5)
    plt.plot(x_, pred2, label='deep', alpha=0.5)
    plt.plot(x_sinc_, y_sinc_, label='sinc(5x*pi)', alpha=0.5)
    plt.legend()
    plt.xlim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('sinc(5x*pi)')
    plt.savefig('hw1-1/{}.png'.format('result'), dpi=600)
    plt.close()
    return

if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (12,6.75)
    x_sinc = np.linspace(-1, 1, 400)
    y_sinc = np.sinc(x_sinc * 5)
    x_sinc, y_sinc = x_sinc[:, np.newaxis], y_sinc[:, np.newaxis]
    net1 = ShallowNet(out_dim=1)
    his1, pred1 = net1.train(x_sinc, y_sinc)
    net2 = DeepNet(out_dim=1)
    his2, pred2 = net2.train(x_sinc, y_sinc)
    plot(his1, his2, 'sinc(5x*pi)', limit=20)
    simulate(pred1, pred2)
    del (net1, net2)