import glob
import json
import re
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb

class Net():
    def __init__(self, data):
        self._load_data(data)
        self._build_net()
        return
    
    def _load_data(self, data_path):
        # self._train_feat = glob.glob(data_path[0])
        # self._test_feat = glob.glob(data_path[2])
        self._train_cap = json.load(open(data_path[1]))
        self._test_cap = json.load(open(data_path[3]))

        # with open(data_path[4], 'r') as f:
        #     self._train_id = [i.rstrip('\n') for i in f.readlines()]
        # with open(data_path[5], 'r') as f:
        #     self._test_id = [i.rstrip('\n') for i in f.readlines()]

        # create the vocabury set
        cap, voc = [], []
        for video in self._train_cap + self._test_cap:
            cap = video['caption'] + cap
        for i in range(len(cap)):
            voc = cap[i].replace('.', '').lower().split() + voc
        self._voc = list(np.unique(voc))[33:-12] + ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self._voc_map = {i: idx for idx, i in enumerate(self._voc)}
        
        # load feature of the video
        # self._train_feat = [np.load(data_path[0]+i+'.npy') for i in self._train_id]
        # self._test_feat = [np.load(data_path[2]+i+'.npy') for i in self._test_id]
        
        # map the feature to the caption
        # for i in self._train_id:

        for i in self._train_cap:
            i['feature'] = np.load(data_path[0]+i['id']+'.npy')
        for i in self._test_cap:
            i['feature'] = np.load(data_path[2]+i['id']+'.npy')

        x_train, y_train, x_test, y_test = [], [], [], []
        

        pdb.set_trace()
        return

    def _build_net(self):
        self._x = tf.placeholder(tf.float32, shape=[None, None])
        
        return

    def _encoder(self, x):
        self._lstm_en = tf.nn.rnn_cell.LSTMCell(num_units=4096)
        return

    def _decoder(self):
        self._lstm_de = tf.nn.rnn_cell.LSTMCell(num_units=len(self._voc))
        return
    
    def _save_result(self):

        return

    def train(self):

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