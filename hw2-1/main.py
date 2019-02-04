import glob
import json
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Net():
    def __init__(self, data):
        self._load_data(data)
        self._build_net()
        return
    
    def _load_data(self, data_path):
        self._train_feat = glob.glob(data_path[0])
        self._test_feat = glob.glob(data_path[2])
        self._train_label = json.load(open(data_path[1]))
        self._test_label = json.load(open(data_path[3]))
        cap = []
        for video in self._train_label:
            cap = video['caption'] + cap
        vab = []
        for i in range(len(cap)):
            vab = cap[i].replace('.', '').lower().split() + vab
        vab = list(np.unique(vab))[33:-12] + ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        print(vab)
        return

    def _build_net(self):
        self._x = tf.placeholder(tf.float32, shape=[None, None])
        
        return

    def _encoder(self, x):
        self._lstm_en = tf.nn.rnn_cell.LSTMCell(num_units=4096)
        return

    def _decoder(self):

        return
    
    def _save_result(self):

        return

    def train(self):

        return
    
    def eval(self):

        return


if __name__ == '__main__':
    TRAIN_PATH = './hw2-1/MLDS_hw2_1_data/training_data/feat/*.npy'
    TRAIN_LABEL = './hw2-1/MLDS_hw2_1_data/training_label.json'
    TEST_PATH = './hw2-1/MLDS_hw2_1_data/testing_data/feat/*.npy'
    TEST_LABEL = './hw2-1/MLDS_hw2_1_data/testing_label.json'
    net = Net((TRAIN_PATH, TRAIN_LABEL, TEST_PATH, TEST_LABEL))
    # train = glob.glob(TRAIN_PATH)[16]
    # print(np.load(train).shape)
    pass