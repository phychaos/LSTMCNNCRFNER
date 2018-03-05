#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.contrib import rnn
import tensorflow as tf


class TextBiLSTM(object):
    def __init__(self, sequence_length, num_classes, word_vocab_size, word_embed_dim, char_vocab_size, grad_clip,
                 num_filters=20, filter_size=3, char_embed_dim=30, n_hidden=200, max_char_per_word=45):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_x_char = tf.placeholder(tf.int32, [None, sequence_length, max_char_per_word], name="input_x_char")
        # 一句话的字符维度 词长度*句长度
        self.input_x_char_flat = tf.reshape(self.input_x_char, [-1, max_char_per_word * sequence_length],
                                            name="input_x_char_flat")

        # 输出标签 B I O
        self.input_y = tf.placeholder(tf.int32, [None, sequence_length], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # 词嵌入层
        with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
            # 词汇量+1(随机词) * 词向量维度
            self.W_word = tf.Variable(tf.random_uniform([word_vocab_size + 1, word_embed_dim], -1, 1), trainable=True,
                                      name="W_word")
            self.word_embedding_placeholder = tf.placeholder(tf.float32, [word_vocab_size + 1, word_embed_dim])
            word_embedding_init = self.W_word.assign(self.word_embedding_placeholder)
            # 输出 数据量 句子长度 向量维度
            self.embedded_words = tf.nn.embedding_lookup(self.W_word, self.input_x, name="embedded_words")

        # 字符嵌入层
        with tf.device('/cpu:0'), tf.name_scope("char_embedding"):
            # 字符数+1(0未知字符) * 字符向量维度
            self.W_char = tf.Variable(tf.random_uniform([char_vocab_size + 1, char_embed_dim], -1, 1), trainable=True,
                                      name="W_char")
            self.char_embedding_placeholder = tf.placeholder(tf.float32, [char_vocab_size + 1, char_embed_dim])
            char_embedding_init = self.W_char.assign(self.char_embedding_placeholder)
            # 输出 数据量 * 字符数*句子长度 * 字符向量维度
            self.embedded_char = tf.nn.embedding_lookup(self.W_char, self.input_x_char_flat, name="embedded_char")
            self.embedded_char_dropout = tf.nn.dropout(self.embedded_char, self.dropout_keep_prob,
                                                       name="embedded_char_dropout")
        # 添加CNN层
        with tf.name_scope("char_conv_maxPool"):
            filter_shape = [filter_size, char_embed_dim, num_filters]
            W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
            b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_conv")

            # will have dimensions [batch_size,out_width,num_filters] out_width is a function of max_words,
            # filter_size and stride_size #(?, 3051, 20)
            conv = tf.nn.conv1d(self.embedded_char_dropout, W_conv, stride=1, padding="SAME", name="conv")

            # out_width for same padding iwth stride 1  given by (max_char_per_word*sequence_length)
            # Apply nonlinearity TODO: Test without relu
            # h = tf.nn.bias_add(conv, b_conv,name="add bias")#does not change dimensions
            h_expand = tf.expand_dims(conv, -1)

            # On the batch size dimension and the channels dimension, ksize is 1
            # because we don't want to take the maximum over multiple examples, or over multiples channels.
            # [batch, height, width, channels]
            pooled = tf.nn.max_pool(h_expand, ksize=[1, sequence_length * max_char_per_word, 1, 1],
                                    strides=[1, max_char_per_word, 1, 1], padding='SAME', name="pooled")

            # print("pooled.get_Shape(): ",pooled.get_shape())
            # [batch_size,(max_char_per_word*sequence_length), num_filters, 1] -->
            # [batch, sequence_length, num_filters] , same as word_embedding layer (?, 113, 20, 1) --> (?, 113, 20)
            self.char_pool_flat = tf.reshape(pooled, [-1, sequence_length, num_filters], name="char_pool_flat")

            # print("self.char_pool_flat.get_shape(): ",self.char_pool_flat.get_shape())
            # [batch, sequence_length, word_embed_dim+num_filters]
            # we mean that the feature with index 2 i/e num_filters is variable
            self.word_char_features = tf.concat([self.embedded_words, self.char_pool_flat], axis=2)

            # print("self.word_char_features.get_shape(): ",self.word_char_features.get_shape())
            self.word_char_features_dropout = tf.nn.dropout(self.word_char_features, self.dropout_keep_prob,
                                                            name="word_char_features_dropout")
        # 添加双向LSTM层
        with tf.name_scope("biLSTM"):
            # LSTM层 前向-后向
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)

            # output : [batch_size, time_steps, cell_fw.output_size]
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                        self.word_char_features_dropout,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)

            self.biLstm = tf.concat([output_fw, output_bw], axis=-1, name="biLstm")
            self.biLstm_clip = tf.clip_by_value(self.biLstm, -grad_clip, grad_clip)
            self.biLstm_dropout = tf.nn.dropout(self.biLstm_clip, self.dropout_keep_prob)
        # 输出层
        with tf.name_scope("output"):
            W_out = tf.get_variable("W_out", shape=[2 * n_hidden, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b_out")
            # [batch_size * time_steps , 2*n_hidden_LSTM] obtained by statement print(self.biLstm.get_shape())
            self.biLstm_reshaped = tf.reshape(self.biLstm_dropout, [-1, 2 * n_hidden])

            # Final (un_normalized) scores and predictions
            # input : [batch_size * time_steps , 2*n_hidden_LSTM] * [2*n_hidden_LSTM, num_classes]  =
            # [batch_size * time_steps , num_classes]
            self.predictions = tf.nn.xw_plus_b(self.biLstm_reshaped, W_out, b_out, name="predictions")
            # output [batch_size, max_seq_len]
            self.logits = tf.reshape(self.predictions, [-1, sequence_length, num_classes], name="logits")

        # 计算交叉熵
        with tf.name_scope("loss"):
            # needs input as  [batch_size, max_seq_len, num_tags]
            # input_y : [batch_size, max_seq_len] 
            # self.logits_clipped = tf.clip_by_value(self.logits,1e-10,10000)
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.input_y,
                                                                                       self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood, name="loss")

        '''with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)'''
