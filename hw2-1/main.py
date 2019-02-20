import glob
import json
import re
import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb
from MLDS_hw2_1_data.bleu_eval import *

n_feat = 4096
n_frame = 80
n_neuron = 512
max_cap_len = 50

class Net():
    def __init__(self, data):
        self._logger = logging.getLogger('Net')
        self._logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
        self._logger.addHandler(ch)

        self._train_data, self._test_data = self._load_data(data)
        self.logits, self.loss, self.summary, self.train_op, self.saver = self._build_net()

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
        self._voc_num = self._voc_num
        
        # map the feature to the caption
        for i in self._train_cap:
            i['feature'] = np.load(data_path[0]+i['id']+'.npy')
        for i in self._test_cap:
            i['feature'] = np.load(data_path[2]+i['id']+'.npy')

        train_data = self._build_dataset(self._train_cap)
        test_data = self._build_dataset(self._test_cap)
        self._n_train = len(self._train_data['x'])

        self._logger.info('Dataset is ready!')
        return train_data, test_data

    def _build_dataset(self, inputs):
        add_unk = lambda x: x if x in self._voc else '<UNK>'
        x, y, cap_len =[], [], []
        for i in inputs:
            for j in i['caption']:
                sentance = ['<BOS>'] + list(map(add_unk, j.rstrip('.').lower().split())) + ['<EOS>']
                cap_len.append(len(sentance))
                y.append([self._voc_map[k] for k in sentance])
                x.append(i['feature'])
        return {'x': np.array(x), 'y': np.array(y), 'cap_len':np.array(cap_len)}

    def _shuffle_data(self):
        choice = np.random.choice(self._n_train, self._n_train)
        self._train_data['x'] = self._train_data['x'][choice]
        self._train_data['y'] = self._train_data['y'][choice]
        self._train_data['cap_len'] = self._train_data['cap_len'][choice]
        return

    def _build_net(self, bs=64):
        # create placeholder for inputs
        with tf.name_scope('inputs'):
            self._x = tf.placeholder(tf.float32, shape=[None, n_frame, n_feat])
            self._y = tf.placeholder(tf.float32, shape=[None, self._voc_num])
            self._sampling = tf.placeholder(tf.bool, [max_cap_len], name='sampling') 
            self._cap_len = tf.placeholder(tf.int32, [None], name='cap_len')
            self._lr = tf.placeholder(tf.float32)
        self._logger.info('placeholders are create.')
        self._bleu = tf.placeholder(tf.float32)

        # create the variables for training
        with tf.name_scope('vars'):
            w_feat = tf.Variable(tf.random_uniform([n_feat, n_neuron], -0.1, 0.1), name='W_feat')
            b_feat = tf.Variable(tf.zeros([n_neuron]), name='b_feat')

            embeddings = tf.Variable(tf.random_uniform([self._voc_num, n_neuron], -0.1, 0.1), name='emb')

            w_de = tf.Variable(tf.random_uniform([n_neuron, self._voc_num], -0.1, 0.1), name='w_dec')
            b_de = tf.Variable(tf.zeros([self._voc_num]), name='b_dec')
        cap_mask = tf.sequence_mask(self._cap_len, max_cap_len, dtype=tf.float32)
        self._logger.info('variables are create.')

        # map the feature into the dimension of encoder
        with tf.name_scope('input_fc'):
            feat_proj = tf.nn.xw_plus_b(tf.reshape(self._x, [-1, n_feat]), w_feat, b_feat)
            feat_reshape = tf.reshape(feat_proj, [-1, n_frame, n_neuron])
            feat = tf.transpose(feat_reshape, perm=[1, 0, 2])

        self._lstm_1 = tf.nn.rnn_cell.LSTMCell(num_units=n_neuron)
        self._lstm_2 = tf.nn.rnn_cell.LSTMCell(num_units=n_neuron)

        state_1 = self._lstm_1.zero_state(bs, dtype=tf.float32)
        state_2 = self._lstm_2.zero_state(bs, dtype=tf.float32)

        pad, h = tf.zeros([bs, n_neuron]), []
        with tf.name_scope('encode_stage'):
            for i in range(n_frame):
                output_1, state_1 = self._lstm_1(feat[i], state_1)
                output_2, state_2 = self._lstm_2(tf.concat([pad, output_1], axis=1), state_2)
                h.append(output_2)

        bos = tf.ones([bs, n_neuron])
        pad_in = tf.zeros([bs, n_neuron])

        logits = []
        max_prob_index = None
        cross_ent_list = []

        for i in range(max_cap_len):
            with tf.name_scope('decode_stage'):
                output_1, state_1 = self._lstm_1(pad_in, state_1)

                if i == 0:
                    output_2, state_2 = self._lstm_2(tf.concat([bos, output_1], axis=1), state_2)
                else:
                    if self._sampling[i] == True:
                        feed_in = self._y[:, i - 1]
                    else:
                        feed_in = tf.argmax(logit_words, 1)

                    with tf.device("/cpu:0"):
                        embed_result = tf.nn.embedding_lookup(embeddings, feed_in)

                    con = tf.concat([embed_result, output_1], axis=1)
                    output_2, state_2 = self._lstm_2(con, state_2)

            with tf.name_scope('output_fc'):
                logit_words =  tf.nn.xw_plus_b(output_2, w_de, b_de)
                logits.append(logit_words)

            with tf.name_scope('cross_entropy'):
                labels = self._y[:, i]
                one_hot_labels = tf.one_hot(labels, self._voc_num, on_value = 1, off_value = None, axis = 1) 
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=one_hot_labels)
                cross_ent_list.append(cross_entropy * cap_mask[:, i])
        

        cross_entropy_tensor = tf.stack(cross_ent_list, 1)
        loss_total = tf.reduce_sum(cross_entropy_tensor, axis=1)
        loss = tf.reduce_mean(loss_total / self._cap_len, axis=0)

        logits_stacked = tf.stack(logits, axis = 0)
        logits_reshaped = tf.reshape(logits_stacked, (max_cap_len, bs, self._voc_num))
        logits = tf.transpose(logits_reshaped, [1, 0, 2])
        
        summary_loss = tf.summary.scalar('training loss', loss)
        summary_bleu = tf.summary.scalar('BLEU score', self._bleu)
        summary = tf.summary.merge_all()
        self._logger.info('network are created.')

        # optimizer and gradient clipping
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self._lr)
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
        
        saver = tf.train.Saver()
        return logits, loss, summary, train_op, saver

    def _encoder(self, x, bs=64):
        self._lstm_1 = tf.nn.rnn_cell.LSTMCell(num_units=n_neuron)
        state = self._lstm_1.zero_state(bs, dtype=tf.float32)
        for i in range(n_frame):
            output, state = self._lstm_1(self._x, state)

        self._logger.info('Build encoder')
        return state

    def _decoder(self, y, en_state):
        self._lstm_de = tf.nn.rnn_cell.LSTMCell(num_units=n_neuron)
        output, final_state = tf.nn.static_rnn(self._lstm_de, y, en_state)
        self._logger.info('Build decoder')
        return
    
    def _save_result(self):

        return

    def train(self, bs, lr=1e-4, epoch=100, save_path='./hw2-1/ckpt/'):
        self._logger.info('training start')
        with tf.Session() as sess:
            for i in range(epoch):
                self._shuffle_data()
                for j in range(self._n_train // bs):
                    fd = {self._x: self._train_data['x'][j*bs: (j+1)*bs],
                        self._y: self._train_data['y'][j*bs: (j+1)*bs],
                        self._cap_len: self._train_data['cap_len'][j*bs: (j+1)*bs],
                        self._sampling: None, self._lr: lr}
                    sess.run(self.train_op, feed_dict=fd)
                
                if i % 10 == 0:
                    self.saver.save(sess, save_path + "model.ckpt")


        return

    def inference(self):

        return
    
    def evaluate(self):

        return


if __name__ == '__main__':
    TRAIN_PATH = './hw2-1/MLDS_hw2_1_data/training_data/feat/'
    TRAIN_LABEL = './hw2-1/MLDS_hw2_1_data/training_label.json'
    TRAIN_ID = './hw2-1/MLDS_hw2_1_data/training_data/id.txt'
    TEST_PATH = './hw2-1/MLDS_hw2_1_data/testing_data/feat/'
    TEST_LABEL = './hw2-1/MLDS_hw2_1_data/testing_label.json'
    TEST_ID = './hw2-1/MLDS_hw2_1_data/testing_data/id.txt'
    net = Net((TRAIN_PATH, TRAIN_LABEL, TEST_PATH, TEST_LABEL, TRAIN_ID, TEST_ID))