import tensorflow as tf
import sklearn
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tqdm
import os
import glob
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

class DeepNet():
    def __init__(self, in_dim=1, out_dim=1, hidden_neuron=0, model_type='regression', name='default'):
        self._name = name
        with tf.name_scope(name):
            self._x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
            self._y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
            self._hidden_layer = tf.layers.dense(self._x, 10+hidden_neuron, activation=tf.nn.relu)
            self._hidden_layer2 = tf.layers.dense(self._hidden_layer, 10+hidden_neuron, activation=tf.nn.relu)
            self._out_layer = tf.layers.dense(self._hidden_layer2, out_dim)
            if model_type=='regression':
                self._loss = tf.losses.mean_squared_error(self._out_layer, self._y)
            else:
                self._loss = tf.losses.softmax_cross_entropy(self._y, self._out_layer)
            self._opt = tf.train.AdamOptimizer()
            self._train_step = self._opt.minimize(self._loss)
            self._acc = tf.equal(tf.argmax(self._y, axis=1), tf.argmax(self._out_layer, axis=1))
        self._pca = sklearn.decomposition.PCA(n_components=2)
        self._list_parameters()
        self._dump_vars(init=True)
        self._count_para()
        return
    
    def train(self, x, y, batch_size=5000, epoch=2000):
        bs = batch_size if batch_size < len(x) else 50
        num_batch, his = len(x) // bs, []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm.tqdm(range(epoch)):
                for j in range(num_batch):
                    fd = {self._x: x[j*bs: (j+1)*bs], self._y: y[j*bs: (j+1)*bs]}
                    sess.run(self._train_step, feed_dict=fd)
                his.append(sess.run(self._loss, feed_dict=fd))
                choice = np.random.choice(len(x), len(x), replace=False)
                x, y = x[choice], y[choice]

                if i % 50 == 0:
                    self._dump_vars(sess=sess, epoch=i, fd=fd)

            pred = sess.run(self._out_layer, feed_dict={self._x: x, self._y: y})

            self._visualize_opti_proc()
            self._visualize_norm()
            self._visualize_loss(his)
        return his, pred, self._count_para()

    def _list_parameters(self):
        for var in tf.trainable_variables():
            print(var.name)
        return

    def _dump_vars(self, init=False, sess=None, epoch=0, fd=None):
        if init:
            weight = [var for var in tf.trainable_variables() if 'kernel' in var.name]
            w_reshape = [tf.reshape(w, [-1]) for w in weight]
            self._w_concat = tf.concat(w_reshape, 0)
            self._norm = tf.add_n([tf.norm(tf.gradients(self._loss, w)) for w in weight])
        else:
            pickle.dump(self._w_concat.eval(), open('hw1-2/weights_%d.p' % epoch, 'wb'))
            pickle.dump(self._norm.eval(feed_dict=fd), open('hw1-2/norm_%d.p' % epoch, 'wb'))
        return

    def _count_para(self):
        return sum([np.prod(list(var.get_shape())) for var in tf.trainable_variables()])

    def _visualize_opti_proc(self):
        weights = np.array([pickle.load(open(f, 'rb')) for f in sorted(glob.glob('hw1-2/weights*.p'))])
        w_pca = self._pca.fit_transform(weights)
        plt.scatter(w_pca[:, 0], w_pca[:, 1])
        for i in range(len(w_pca)):
            plt.text(w_pca[i, 0], w_pca[i, 1]+0.2, str(50*i))
        plt.title('Optimization Process')
        plt.xlabel('PC 0')
        plt.ylabel('PC 1')
        plt.savefig('hw1-2/opti_proc.png', dpi=500)
        plt.close()
        return
    
    def _visualize_norm(self):
        norms = np.array([pickle.load(open(f, 'rb')) for f in glob.glob('hw1-2/norm*.p')])
        plt.plot(range(len(norms)), norms)
        plt.title('Gradient Norm During Training')
        plt.yscale('log')
        plt.xlabel('epoches')
        plt.ylabel('2-norm')
        plt.savefig('hw1-2/norm.png', dpi=500)
        plt.close()
        return

    def _visualize_loss(self, loss):
        plt.plot(range(len(loss)), loss)
        plt.yscale('log')
        plt.xlabel('epoches')
        plt.ylabel('loss')
        plt.savefig('hw1-2/loss.png', dpi=500)
        plt.close()
        return

if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (12,6.75)
    (x_mnist, y_), (_, __) = tf.keras.datasets.mnist.load_data()
    x_mnist, y_mnist = x_mnist.reshape([x_mnist.shape[0], -1]), np.zeros((len(x_mnist), 10))
    y_mnist[np.arange(60000), y_] = 1
    para = []
    for i in np.arange(100):
        net2 = DeepNet(in_dim=28*28, out_dim=10, hidden_neuron=i, model_type='classification')
        his2, _, num_para = net2.train(x_mnist, y_mnist, epoch=2000)
        para.append(num_para)