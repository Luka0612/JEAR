# coding: utf-8
from __future__ import division

import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _luong_score

# a = np.ones(shape=(2, 10), dtype=np.float32)
# b = np.ones(shape=(2, 4, 10), dtype=np.float32)
# w = np.ones(shape=(4, 7), dtype=np.float32)
#
# a1 = tf.constant(a, dtype=tf.float32)
# b1 = tf.constant(b, dtype=tf.float32)
# w1 = tf.constant(w, dtype=tf.float32)
#
# # c = _luong_score(a1, b1, True)
# a1 = array_ops.expand_dims(a1, 1)
# score = math_ops.matmul(a1, b1, transpose_b=True)
# score = array_ops.squeeze(score, [1])

# a = np.array([[1, 2, 13, 19], [3, 4, 14, 19]], dtype=np.float32)
# b = np.array([[5, 6, 15, 19], [7, 8, 16, 19]], dtype=np.float32)
# c = np.array([[9, 10, 17, 19], [11, 12, 18, 19]], dtype=np.float32)
#
# a = tf.constant(a, dtype=tf.float32)
# a = array_ops.expand_dims(a, 1)
# b = tf.constant(b, dtype=tf.float32)
# b = array_ops.expand_dims(b, 1)
# c = tf.constant(c, dtype=tf.float32)
# c = array_ops.expand_dims(c, 1)
#
# con = tf.concat([a, b, c], 1)
#
# s = tf.transpose(con, perm=[0, 2, 1])

# a = np.array([[[1, 2], [13, 19]], [[3, 4], [14, 19]]], dtype=np.float32)
# a = tf.constant(a, dtype=tf.float32)
#
# b = np.array([[[5, 6], [15, 19]], [[7, 8], [16, 19]]], dtype=np.float32)
# b = tf.constant(b, dtype=tf.float32)
#
# con = tf.concat([a, b], 1)


masks = tf.sequence_mask([5, 4], 10, dtype=tf.float32, name="masks")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print sess.run(masks)
print sess.run(masks).shape

