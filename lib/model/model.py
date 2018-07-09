# coding: utf-8
from __future__ import division

import os
import sys

import tensorflow as tf

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))
sys.path.append(os.path.abspath("../utils"))
from utils import generate_batch

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_config = tf.ConfigProto()
# 指定内存占用
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.95


class Model():
    def __init__(self, is_training, config):
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, config.num_steps])
        self.train_entity_labels = tf.placeholder(tf.float32, shape=[None, config.num_steps, config.entity_class_num])

        if is_training:
            # 词向量
            with tf.variable_scope("model", reuse=None):
                embeddings = tf.get_variable(
                    "embedding", [config.vocabulary_size, config.embedding_size], dtype=tf.float32)
        else:
            with tf.variable_scope("model", reuse=True):
                embeddings = tf.get_variable(
                    "embedding", [config.vocabulary_size, config.embedding_size])

        # [0] -> all 0
        one_hot = tf.one_hot(0, config.vocabulary_size, dtype=tf.float32)
        one_hot = tf.transpose(tf.reshape(tf.tile(one_hot, [config.embedding_size]),
                                          shape=[config.embedding_size, config.vocabulary_size]))
        embeddings = embeddings - embeddings[0] * one_hot

        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        if is_training:
            embed = tf.nn.dropout(embed, config.keep_prob)

        with tf.variable_scope('encoder'):
            encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_a_cell(config.encode_hidden_size, is_training, config.keep_prob) for _ in
                 range(config.encode_num_layers)])
            encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_a_cell(config.encode_hidden_size, is_training, config.keep_prob) for _ in
                 range(config.encode_num_layers)])
            # 初始状态
            batch_size = tf.shape(self.train_inputs)[0]
            encoder_cell_fw_initial_state = encoder_cell_fw.zero_state(batch_size, dtype=tf.float32)
            encoder_cell_bw_initial_state = encoder_cell_bw.zero_state(batch_size, dtype=tf.float32)

            length = get_length(embed)

            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw, encoder_cell_bw,
                                                                             embed,
                                                                             sequence_length=length,
                                                                             initial_state_fw=encoder_cell_fw_initial_state,
                                                                             initial_state_bw=encoder_cell_bw_initial_state)
            encoder_outputs = tf.concat(encoder_outputs, 2)

            # encoder_state = [tf.concat(last_state, 2) for last_state in encoder_state]
            # encoder_state = tuple(
            #     [LSTMStateTuple(last_states_concat[0], last_states_concat[1]) for last_states_concat in
            #      encoder_state])

        # embedding layer
        with tf.variable_scope('target_embedding'):
            if is_training:
                with tf.variable_scope("target_model", reuse=None):
                    decoder_embedding = tf.get_variable(
                        "target_embedding", [config.entity_class_num, config.decode_embedding_dim], dtype=tf.float32)
            else:
                with tf.variable_scope("target_model", reuse=True):
                    decoder_embedding = tf.get_variable(
                        "target_embedding", [config.entity_class_num, config.decode_embedding_dim])

            target_input_embeddeds = tf.nn.embedding_lookup(decoder_embedding, self.train_entity_labels)

        if False:
            with tf.variable_scope('decode'):
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self.get_a_cell(config.decode_hidden_size, is_training, config.keep_prob) for _ in
                     range(config.decode_num_layers)])
                decoder_cell_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)
                outputs = list()
                state = decoder_cell_initial_state
                # 全连接层
                softmax_w = tf.get_variable(
                    "softmax_w", [config.decode_hidden_size, config.entity_class_num], dtype=tf.float32)
                softmax_bias = tf.get_variable("softmax_b", [config.entity_class_num], dtype=tf.float32)
                with tf.variable_scope('RNN'):
                    for timestep in range(length):
                        if timestep > 0:
                            tf.get_variable_scope().reuse_variables()
                            if is_training:
                                decode_input = tf.concat(2, [encoder_outputs[:, timestep, :],
                                                             target_input_embeddeds[:, timestep, :]])
                            else:
                                last_label = tf.argmax(tf.matmul(outputs[-1], softmax_w) + softmax_bias, 1)
                                last_label_embeddeds = tf.nn.embedding_lookup(decoder_embedding, last_label)
                                decode_input = tf.concat(2, [encoder_outputs[:, timestep, :],
                                                             last_label_embeddeds])
                            (cell_output, state) = decoder_cell(decode_input, state)
                            outputs.append(cell_output)
                output = tf.reshape(tf.concat(1, outputs), [-1, config.decode_hidden_size])
                logits = tf.matmul(output, softmax_w) + softmax_bias

                # loss , shape=[batch*num_steps]
                # 带权重的交叉熵计算
                loss = tf.nn.seq2seq.sequence_loss_by_example(
                    [logits],  # output [batch*numsteps, vocab_size]
                    [tf.reshape(self.train_entity_labels, [-1])],  # target, [batch_size, num_steps] 然后展开成一维【列表】
                    [tf.ones([batch_size * length], dtype=tf.float32)])  # weight

        self._lr = tf.Variable(1.0, trainable=False)
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")  # 用于外部向graph输入新的 lr值
        self._lr_update = tf.assign(self._lr, self._new_lr)  # 使用new_lr来更新lr的值

    def assign_lr(self, session, lr_value):
        # 使用 session 来调用 lr_update 操作
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def lr(self):
        return self._lr

    def get_a_cell(self, lstm_size, is_training, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        if is_training and keep_prob < 1:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)
        return lstm


def get_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    sequence_length = tf.reduce_sum(used, 1)
    sequence_length = tf.cast(sequence_length, tf.int32)
    return sequence_length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def model_save(sess, path, model_name, global_step):
    # 模型保存
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(path, model_name), global_step=global_step)
    return saver


class TrainConfig(object):
    # 词向量维度
    num_steps = 128
    # batch的大小
    batch_size = 8
    entity_class_num = 18
    vocabulary_size = 418129
    embedding_size = 128
    keep_prob = 0.9
    encode_hidden_size = 128
    encode_num_layers = 4
    decode_embedding_dim = 128
    decode_hidden_size = 128
    decode_num_layers = 4
    max_max_epoch = 1000
    # 学习率
    lr = 0.8
    # 学习率衰减
    lr_decay = 0.8
    max_epoch = 5
    datafile = current_relative_path("../../data/corpus_prepared.pickled")
    wordvectors = current_relative_path("../../data/vecs.lc.over100freq.txt.gz")
    contextsize = 5
    start_split_data_index = 0.0
    end_split_data_index = 0.8


class TestConfig(object):
    # 词向量维度
    num_steps = 128
    # batch的大小
    batch_size = 8
    entity_class_num = 18
    vocabulary_size = 418129
    embedding_size = 128
    keep_prob = 0.9
    encode_hidden_size = 128
    encode_num_layers = 4
    decode_embedding_dim = 128
    decode_hidden_size = 128
    decode_num_layers = 4
    start_split_data_index = 0.8
    end_split_data_index = 1.0


def run_epoch(session, model, data):
    accuracys = 0.0
    costs = 0.0
    iters = 0
    batch_inputs, batch_labels = data.next()
    while batch_inputs is not None:
        accuracy, cost, _ = session.run([model.lr], feed_dict={model.train_inputs: batch_inputs,
                                                               model.train_entity_labels: batch_labels})
        batch_inputs, batch_labels = data.next()
    return accuracys / iters, costs / iters


def train():
    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            # 训练模型， is_trainable=True
            m = Model(is_training=True, config=TrainConfig)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            # 测试模型
            mtest = Model(is_training=False, config=TestConfig)

        session.run(tf.global_variables_initializer())
        for i in range(TrainConfig.max_max_epoch):
            lr_decay = TrainConfig.lr_decay ** (i // TrainConfig.max_epoch)
            lr = TrainConfig.lr * lr_decay
            m.assign_lr(session, lr)

            train_data = generate_batch(TrainConfig)
            train_accuracy, train_loss = run_epoch(session, m, train_data)
            print "Epoch: %d Train accuracy: %.3f, Train loss: %.3f, with lr: %.3f" % (
                i + 1, train_accuracy, train_loss, lr)

            if i % 10 == 0:
                test_data = generate_batch(TestConfig)
                test_accuracy, test_loss = run_epoch(session, mtest, test_data)
                print "Epoch: %d Test accuracy: %.3f, Test loss: %.3f" % (i, test_accuracy, test_loss)