import tensorflow as tf
from tensorflow.keras import backend as K
import glob
import random
import pretty_midi
import IPython
import numpy as np
from tqdm import tnrange, tqdm_notebook, tqdm
from random import shuffle, seed
import numpy as np
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Nadam
import numpy as np
from numpy.random import choice
import pickle
import matplotlib.pyplot as plt

import unicodedata
import re
import numpy as np
import os
import io
import time


class Model:

  def __init__(self,
               seq_len,
               unique_notes,
               dropout=0.3,
               output_emb=100,
               rnn_unit=128,
               dense_unit=64):

    self.seq_len = seq_len
    self.unique_notes = unique_notes
    self.dropout = dropout
    self.output_emb = output_emb
    self.rnn_unit = rnn_unit
    self.dense_unit = dense_unit

  def placeholders(self):
    with tf.name_scope('input'):
      image_pl = tf.placeholder(tf.float32, (None, self.seq_len), name="input")
      label_pl = tf.placeholder(tf.int32, (None, 1), name="label")

    return image_pl, label_pl

  def loss(self, pred, labels):
    with tf.name_scope('loss'):
      loss = tf.losses.sparse_softmax_cross_entropy(labels, pred)
    return loss

  def optimizer(self, loss):
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,
                                           momentum=0.9)
    return optimizer.minimize(loss)

  def infer(self, inputs):
    embedding = tf.keras.layers.Embedding(input_dim=self.unique_notes + 1,
                                          output_dim=self.output_emb,
                                          input_length=self.seq_len)(inputs)

    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(self.rnn_unit, return_sequences=True))(embedding)


    forward_pass = tf.keras.layers.Dropout(self.dropout)(forward_pass)

    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(self.rnn_unit, return_sequences=True))(forward_pass)


    forward_pass = tf.keras.layers.Dropout(self.dropout)(forward_pass)

    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(self.rnn_unit))(forward_pass)

    forward_pass = tf.keras.layers.Dropout(self.dropout)(forward_pass)

    forward_pass = tf.keras.layers.Dense(self.dense_unit)(forward_pass)

    forward_pass = tf.keras.layers.LeakyReLU()(forward_pass)

    outputs = tf.keras.layers.Dense(self.unique_notes + 1, activation=None)(forward_pass)

    return outputs