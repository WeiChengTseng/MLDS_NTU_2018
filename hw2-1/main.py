import glob
import json
import re
import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
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

        self._logger.info('Net __init__')
        self._train_data, self._test_data = self._load_data(data)
        self.logits, self.loss, self.summary, self.train_op, self.saver = self._build_net()

        return
    
    def _load_data(self, data_path, is_reload=False):
        if is_reload:
            train_data = pickle.load(open('./hw2-1/train_data.p', 'rb'))
            test_data = pickle.load(open('./hw2-1/test_data.p', 'rb'))
            self._logger.info('Dataset is reloaded.')
            return train_data, test_data

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
        self._voc_num = len(self._voc)
        
        # map the feature to the caption
        for i in self._train_cap:
            i['feature'] = np.load(data_path[0]+i['id']+'.npy')
        for i in self._test_cap:
            i['feature'] = np.load(data_path[2]+i['id']+'.npy')

        train_data = self._build_dataset(self._train_cap)
        test_data = self._build_dataset(self._test_cap)
        self._n_train = len(train_data['x'])
        self._n_test = len(test_data['x'])

        self._logger.info('Dataset is built!')
        # pickle.dump(train_data, open('./hw2-1/train_data.p', 'wb'))
        # pickle.dump(test_data, open('./hw2-1/test_data.p', 'wb'))
        # self._logger.info('Dataset is saved.')
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
        # pdb.set_trace()
        return {'x': np.array(x), 'y': np.array(y), 'cap_len':np.array(cap_len)}

    def _shuffle_data(self):
        choice = np.random.choice(self._n_train, self._n_train)
        self._train_data['x'] = self._train_data['x'][choice]
        self._train_data['y'] = self._train_data['y'][choice]
        self._train_data['cap_len'] = self._train_data['cap_len'][choice]

        choice = np.random.choice(self._n_test, self._n_test)
        self._test_data['x'] = self._test_data['x'][choice]
        self._test_data['y'] = self._test_data['y'][choice]
        self._test_data['cap_len'] = self._test_data['cap_len'][choice]
        return

    def _build_net(self, bs=64):
        # create placeholder for inputs
        with tf.name_scope('inputs'):
            self._x = tf.placeholder(tf.float32, shape=[None, n_frame, n_feat])
            self._y = tf.placeholder(tf.uint8, shape=[None, max_cap_len])
            # self._sampling = tf.placeholder(tf.bool, [max_cap_len], name='sampling') 
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

        # encoding stage
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

        # decoding stage
        with tf.name_scope('decode_stage'):
            for i in range(max_cap_len):
                output_1, state_1 = self._lstm_1(pad_in, state_1)
                if i == 0:
                    output_2, state_2 = self._lstm_2(tf.concat([bos, output_1], axis=1), state_2)
                else:
                    # if self._sampling[i] == True:
                    #     feed_in = self._y[:, i-1]
                    # else:
                    #     feed_in = tf.argmax(logit_words, 1)
                    feed_in = tf.argmax(logit_words, 1)

                    with tf.device("/cpu:0"):
                        embed_result = tf.nn.embedding_lookup(embeddings, feed_in)
                    con = tf.concat([embed_result, output_1], axis=1)
                    output_2, state_2 = self._lstm_2(con, state_2)

            # with tf.name_scope('output_fc'):
                logit_words =  tf.nn.xw_plus_b(output_2, w_de, b_de)
                logits.append(logit_words)

            
            labels = self._y[:, i]
            one_hot_labels = tf.one_hot(labels, self._voc_num, on_value = 1, off_value = None, axis = 1) 
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=one_hot_labels)
            cross_ent_list.append(cross_entropy * cap_mask[:, i])
        
        # compute loss
        with tf.name_scope('cross_entropy'):
            cross_entropy_tensor = tf.stack(cross_ent_list, 1)
            loss_total = tf.reduce_sum(cross_entropy_tensor, axis=1)
            loss = tf.reduce_mean(loss_total / tf.cast(self._cap_len, tf.float32), axis=0)

        logits_stacked = tf.stack(logits, axis = 0)
        logits_reshaped = tf.reshape(logits_stacked, (max_cap_len, bs, self._voc_num))
        logits = tf.transpose(logits_reshaped, [1, 0, 2])
        
        # tensorflow summary
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

    def _compute_bleu(self, mode='train', out_file='./hw2-1/output/output.txt'):
        result, bleu = {}, []
        with open(out_file,'r') as f:
            for line in f:
                line = line.rstrip()
                comma = line.index(',')
                test_id = line[:comma]
                caption = line[comma+1:]
                result[test_id] = caption

        cap_label = self._train_cap if mode == 'train' else self._test_cap
        for item in cap_label:
            score_per_video = []
            captions = [x.rstrip('.') for x in item['caption']]
            score_per_video.append(BLEU(result[item['id']],captions,True))
            bleu.append(score_per_video[0])
        average = sum(bleu) / len(bleu)
        self._logger.info("Average bleu score = " + str(average))
        return average

    def _tensorflow_log(self, sess, writer, bs, step):
        fd = {self._x: self._train_data['x'][ :bs],
            self._y: self._train_data['y'][ :bs],
            self._cap_len: self._train_data['cap_len'][ :bs],}
            # self._sampling: None}
        summ_train = sess.run(self.summary, feed_dict=fd)
        writer['train'].add_summary(summ_train, step)  

        fd = {self._x: self._test_data['x'][ :bs],
                self._y: self._test_data['y'][: bs],
                self._cap_len: self._test_data['cap_len'][: bs],
                self._bleu: self._compute_bleu(mode='test'),}
                # self._sampling: None }
        summ_test = sess.run(self.summary, feed_dict=fd)
        writer['test'].add_summary(summ_test, step)
        return

    def train(self, bs=64, lr=1e-4, epoch=100, save_path='./hw2-1/ckpt/'):
        self._logger.info('training start')

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('./hw2-1/logs/train/', sess.graph)
            test_writer = tf.summary.FileWriter('./hw2-1/logs/test/')
            writer = {'train': train_writer, 'test': test_writer}
            self._logger.info('writer is created.')
            # return
            for i in range(epoch):
                self._shuffle_data()
                for j in range(self._n_train // bs):
                    fd = {self._x: self._train_data['x'][j*bs: (j+1)*bs],
                        self._y: self._train_data['y'][j*bs: (j+1)*bs],
                        self._cap_len: self._train_data['cap_len'][j*bs: (j+1)*bs],
                        # self._sampling: None, 
                        self._lr: lr}
                    sess.run(self.train_op, feed_dict=fd)

                self._tensorflow_log(sess, writer, bs, i)
                
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
    net.train()
    net.evaluate()