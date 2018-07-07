# coding: utf-8
from __future__ import division

import os

import tensorflow as tf

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_config = tf.ConfigProto()
# 指定内存占用
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.95


class Model():
    def __init__(self, is_training, config):
        pass
