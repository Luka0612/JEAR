# coding: utf-8
from __future__ import division

import os
import sys

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _luong_score
from tensorflow.python.ops import array_ops

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))
sys.path.append(os.path.abspath(".."))
from data_process.generate_data_stream import generate_batch

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_config = tf.ConfigProto()
# 指定内存占用
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.95


class Model():
    def __init__(self, is_training, config):
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, config.num_steps])
        self.train_entity_labels = tf.placeholder(tf.int32, shape=[None, config.num_steps, config.entity_class_num])
        self.train_entity_labels_index = tf.placeholder(tf.int32, shape=[None, config.num_steps])

        self.train_rel_entity_index = tf.placeholder(tf.int32, shape=[None, config.rel_entity_index_num])
        self.train_rel_entity_index_match = tf.placeholder(tf.int32, shape=[None, config.rel_entity_index_num,
                                                                            config.entity_rel_num])
        self.train_rel = tf.placeholder(tf.int32, shape=[None, config.rel_entity_index_num,
                                                         config.entity_rel_num])
        self.rel_masks = tf.placeholder(tf.int32, shape=[None, config.rel_entity_index_num])
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

            target_input_embeddeds = tf.nn.embedding_lookup(decoder_embedding, self.train_entity_labels_index)

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
                for timestep in range(config.num_steps):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    if is_training:
                        decode_input = tf.concat([encoder_outputs[:, timestep, :],
                                                  target_input_embeddeds[:, timestep, :]], 1)
                    else:
                        if timestep == 0:
                            last_label_embeddeds = tf.nn.embedding_lookup(decoder_embedding, [0] * config.batch_size)
                        else:
                            last_label = tf.argmax(tf.matmul(outputs[-1], softmax_w) + softmax_bias, 1)
                            last_label_embeddeds = tf.nn.embedding_lookup(decoder_embedding, last_label)
                        decode_input = tf.concat([encoder_outputs[:, timestep, :],
                                                  last_label_embeddeds], 1)
                    (cell_output, state) = decoder_cell(decode_input, state)
                    outputs.append(cell_output)
            output = tf.reshape(tf.concat(outputs, 1), [-1, config.decode_hidden_size])
            logits = tf.reshape(tf.matmul(output, softmax_w) + softmax_bias,
                                [batch_size, config.num_steps, config.entity_class_num])

            masks = tf.sequence_mask(length, config.num_steps, dtype=tf.float32, name="masks")
            cost_entity = tf.contrib.seq2seq.sequence_loss(logits, self.train_entity_labels_index, masks)

        with tf.variable_scope('rel_decode'):
            # 转向关系空间
            label_embed = tf.concat([encoder_outputs, target_input_embeddeds], 2)
            label_embed = tf.reshape(tf.pad(label_embed, paddings=[[0, 0], [0, 1], [0, 0]], mode="CONSTANT"),
                                     [-1, config.decode_embedding_dim + 2 * config.encode_hidden_size])

            rel_class_w = tf.get_variable(
                "rel_class_w", [config.decode_embedding_dim + 2 * config.encode_hidden_size, config.rel_class_w],
                dtype=tf.float32)
            rel_class_b = tf.get_variable("rel_class_b", [config.rel_class_w], dtype=tf.float32)
            label_embed = tf.reshape(tf.nn.relu(tf.matmul(label_embed, rel_class_w) + rel_class_b),
                                     [batch_size, -1, config.rel_class_w])

            # lookup
            train_rel_entity_index_embed = []
            for i in range(config.batch_size):
                train_rel_entity_index_embed.append(
                    tf.nn.embedding_lookup(label_embed[i], self.train_rel_entity_index[i]))
            train_rel_entity_index_embed = tf.reshape(tf.concat([train_rel_entity_index_embed], 0),
                                                      [batch_size, config.rel_entity_index_num, config.rel_class_num,
                                                       -1])

            train_rel_entity_index_match_embed = []
            for i in range(config.batch_size):
                train_rel_entity_index_match_embed_batch = []
                for j in range(config.rel_entity_index_num):
                    train_rel_entity_index_match_embed_batch.append(
                        tf.nn.embedding_lookup(label_embed[i], self.train_rel_entity_index_match[i][j]))

                train_rel_entity_index_match_embed.append(tf.concat([train_rel_entity_index_match_embed_batch], 0))
            train_rel_entity_index_match_embed = tf.reshape(tf.concat([train_rel_entity_index_match_embed], 0),
                                                            [batch_size, config.rel_entity_index_num,
                                                             config.entity_rel_num, config.rel_class_num, -1])

            # 采用LuongAttention
            cost_rels = []
            for timestep in range(config.rel_entity_index_num):
                attention_score_timestep = []
                for rel_class_index in range(config.rel_class_num):
                    sub_score = _luong_score(train_rel_entity_index_embed[:, timestep, rel_class_index, :],
                                             train_rel_entity_index_match_embed[:, timestep, :,
                                             rel_class_index, :], False)
                    sub_score = array_ops.expand_dims(sub_score, 1)
                    attention_score_timestep.append(sub_score)
                attention_score_timestep = tf.transpose(tf.concat(attention_score_timestep, 1), perm=[0, 2, 1])
                masks = tf.sequence_mask(self.rel_masks[:, timestep], config.entity_rel_num, dtype=tf.float32,
                                         name="rel_masks")
                cost_rel = tf.contrib.seq2seq.sequence_loss(attention_score_timestep, self.train_rel[:, timestep, :],
                                                            masks)
                cost_rels.append(cost_rel)
            self.cost = tf.reduce_mean(cost_rels) + cost_entity

        self._lr = tf.Variable(1.0, trainable=False)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        self.optimizer = self.clip_gradients(optimizer, self.cost, config.max_grad_norm)
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")  # 用于外部向graph输入新的 lr值
        self._lr_update = tf.assign(self._lr, self._new_lr)  # 使用new_lr来更新lr的值

    def assign_lr(self, session, lr_value):
        # 使用 session 来调用 lr_update 操作
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def clip_gradients(self, optimizer, loss, max_grad_norm):
        gradients, v = zip(*optimizer.compute_gradients(loss))
        # 为了避免梯度爆炸的问题，我们求出梯度的二范数。
        # 然后判断该二范数是否大于1.25，若大于，则变成
        # gradients * (1.25 / global_norm)作为当前的gradients
        gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
        # 将刚刚求得的梯度组装成相应的梯度下降法
        optimizer = optimizer.apply_gradients(zip(gradients, v))
        return optimizer

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
    num_steps = 120
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
    contextsize = 120
    start_split_data_index = 0.0
    end_split_data_index = 0.8
    rel_entity_index_num = 31
    entity_rel_num = 30
    rel_attention_w = 128
    rel_class_num = 7
    rel_class_w = 64 * 7
    max_grad_norm = 5


class TrainTestConfig(object):
    # 词向量维度
    num_steps = 120
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
    max_epoch = 1
    datafile = current_relative_path("../../data/corpus_prepared.pickled")
    wordvectors = current_relative_path("../../data/vecs.lc.over100freq.txt.gz")
    contextsize = 120
    start_split_data_index = 0.8
    end_split_data_index = 1.0
    rel_entity_index_num = 31
    entity_rel_num = 30
    rel_attention_w = 128
    rel_class_num = 7
    rel_class_w = 64 * 7
    max_grad_norm = 5


def run_epoch(session, model, data):
    costs = 0.0
    iters = 0
    batch_inputs, batch_labels, batch_labels_index, batch_rel_entity_index, batch_rel_entity_index_match, train_rel, rel_masks = data.next()
    while batch_inputs is not None:
        cost, _ = session.run([model.cost, model.optimizer], feed_dict={model.train_inputs: batch_inputs,
                                                                        model.train_entity_labels: batch_labels,
                                                                        model.train_entity_labels_index: batch_labels_index,
                                                                        model.train_rel_entity_index: batch_rel_entity_index,
                                                                        model.train_rel_entity_index_match: batch_rel_entity_index_match,
                                                                        model.train_rel: train_rel,
                                                                        model.rel_masks: rel_masks})
        print iters, ":", cost
        costs += cost
        iters += 1
        batch_inputs, batch_labels, batch_labels_index, batch_rel_entity_index, batch_rel_entity_index_match, train_rel, rel_masks = data.next()
    return costs / iters


def train():
    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            # 训练模型， is_trainable=True
            m = Model(is_training=True, config=TrainConfig)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            # 测试模型
            mtest = Model(is_training=False, config=TrainTestConfig)
            mtest.optimizer = tf.no_op()

        session.run(tf.global_variables_initializer())
        for i in range(TrainConfig.max_max_epoch):
            lr_decay = TrainConfig.lr_decay ** (i // TrainConfig.max_epoch)
            lr = TrainConfig.lr * lr_decay
            m.assign_lr(session, lr)

            train_data = generate_batch(TrainConfig)
            train_loss = run_epoch(session, m, train_data)
            print "Epoch: %d Train loss: %.3f, with lr: %.3f" % (i + 1, train_loss, lr)

            if i % 10 == 0:
                test_data = generate_batch(TrainTestConfig)
                test_loss = run_epoch(session, mtest, test_data)
                print "Epoch: %d Test loss: %.3f" % (i + 1, test_loss)


if __name__ == '__main__':
    train()
