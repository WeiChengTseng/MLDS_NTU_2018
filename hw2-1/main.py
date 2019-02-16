import glob
import json
import re
import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb

class Net():
    def __init__(self, data):
        self._logger = logging.getLogger('Net')
        self._logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
        self._logger.addHandler(ch)

        self._x_train, self._y_train, self._x_test, self._y_test = self._load_data(data)
        pdb.set_trace()
        self._build_net()
        return
    
    def _load_data(self, data_path):
        self._train_cap = json.load(open(data_path[1]))
        self._test_cap = json.load(open(data_path[3]))

        # create the vocabury set
        cap, voc = [], []
        for video in self._train_cap + self._test_cap:
            cap = video['caption'] + cap
        for i in range(len(cap)):
            voc = cap[i].replace('.', '').lower().split() + voc
        self._voc = list(np.unique(voc))[32:-12] + ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self._voc_map = {i: idx for idx, i in enumerate(self._voc)}
        
        # map the feature to the caption
        for i in self._train_cap:
            i['feature'] = np.load(data_path[0]+i['id']+'.npy')
        for i in self._test_cap:
            i['feature'] = np.load(data_path[2]+i['id']+'.npy')

        x_train, y_train = self._build_dataset(self._train_cap)
        x_test, y_test = self._build_dataset(self._test_cap)

        self._logger.info('Dataset is ready!')
        return x_train, y_train, x_test, y_test

    def _build_dataset(self, cap):
        add_unk = lambda x: x if x in self._voc else '<UNK>'
        x, y =[], []
        for i in cap:
            for j in i['caption']:
                sentance = ['<BOS>'] + list(map(add_unk, j.rstrip('.').lower().split())) + ['<EOS>']
                one_hot = np.zeros((len(sentance), len(self._voc)))
                one_hot[np.arange(len(sentance)), [self._voc_map[k] for k in sentance]] = 1
                y.append(one_hot)
                x.append(i['feature'])
        # pdb.set_trace()
        return np.array(x), np.array(y)

    def _build_net(self):
        self._x = tf.placeholder(tf.float32, shape=[None, None, None])
        self._y = tf.placeholder(tf.float32, shape=[None, None])
        target_seq_len = tf.placeholder(tf.int32, [None], name='target_seq_len')
        max_target_len = tf.reduce_max(target_seq_len)  

        encode_state = self._encoder(self._x)
        self._decoder(self._y, encode_state)

        lr, cost = None, None
        optimizer = tf.train.AdamOptimizer(lr)
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        
        return

    def _encoder(self, x, bs=64):
        self._lstm_en = tf.nn.rnn_cell.LSTMCell(num_units=512)
        init_state = self._lstm_en.zero_state(bs, dtype=tf.float32)
        output, final_state = tf.nn.static_rnn(self._lstm_en, x, init_state)

        self._logger.info('Build encoder')
        return final_state

    def _decoder(self, y, en_state):
        self._lstm_de = tf.nn.rnn_cell.LSTMCell(num_units=512)
        output, final_state = tf.nn.static_rnn(self._lstm_de, y, en_state)
        self._logger.info('Build decoder')
        return
    
    def _save_result(self):

        return

    def train(self, bs):
        self._logger.info('training')

        with tf.Session() as sess:
            pass

        return
    
    def eval(self):

        return


if __name__ == '__main__':
    # TRAIN_PATH = './hw2-1/MLDS_hw2_1_data/training_data/feat/*.npy'
    TRAIN_PATH = './hw2-1/MLDS_hw2_1_data/training_data/feat/'
    TRAIN_LABEL = './hw2-1/MLDS_hw2_1_data/training_label.json'
    TRAIN_ID = './hw2-1/MLDS_hw2_1_data/training_data/id.txt'
    TEST_PATH = './hw2-1/MLDS_hw2_1_data/testing_data/feat/'
    TEST_LABEL = './hw2-1/MLDS_hw2_1_data/testing_label.json'
    TEST_ID = './hw2-1/MLDS_hw2_1_data/testing_data/id.txt'
    net = Net((TRAIN_PATH, TRAIN_LABEL, TEST_PATH, TEST_LABEL, TRAIN_ID, TEST_ID))
    # train = glob.glob(TRAIN_PATH)[16]
    # print(np.load(train).shape)
    pass