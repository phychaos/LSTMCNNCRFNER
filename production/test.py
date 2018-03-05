#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from config.parameter import *
from core import utils
from production import data_processor as dp
# @File: train.py
# @Time: 18-2-5 下午4:14
from production.data_pickle import load_data_dict
from production.network import TextBiLSTM

from LSTMCNNCRFNER.config.config import *
from LSTMCNNCRFNER.production import aux_network_func as af

flags_dict, word_alphabet, label_alphabet, char_alphabet = load_data_dict()

# 加载词向量 词向量维度 word2vec glove
embed_dict, embed_dim, case_less = utils.load_word_embedding_dict(embedding, embedding_path)
# 为词分配向量值, 若不存在于glove则分配随机值 词向量大小*维度
embed_table = dp.build_embed_table(word_alphabet, embed_dict, embed_dim, case_less)

max_length = flags_dict.get('max_length', MAX_LENGTH)
num_classes = flags_dict['num_classes']
word_vocab_size = flags_dict['word_vocab_size']
char_vocab_size = flags_dict['char_vocab_size']
max_char_per_word = flags_dict['max_char_per_word']
char_embed_table = flags_dict['char_embed_table']
checkpoint_dir = flags_dict['checkpoint_dir']


def test():
    print("读取测试集...")
    # 读取验证集
    x_word_test, y_label_test, x_test, y_test = dp.read_sequence_labeling(
        TEST_NER_PATH, word_alphabet, label_alphabet, word_column, label_column)

    # 验证集-标签 数据补全 补全数据集 实际数据长度
    x_test_pad, test_seq_length = utils.pad_sequence(x_test, max_length, PadZeroBegin)
    y_test_pad, _ = utils.pad_sequence(y_test, max_length, PadZeroBegin)
    # 从验证集创建字符集-索引
    char_index_test, _ = dp.generate_character_data(x_word_test, char_alphabet)

    # 构建字符-id 数据集 数据长度 * 句子长度 * 词汇长度
    char_index_test_pad = dp.construct_padded_char(char_index_test, char_alphabet, max_length, max_char_per_word)
    test_batches = list(utils.test_batch_iter(
        list(zip(x_test_pad, y_test_pad, test_seq_length, char_index_test_pad, x_word_test)),
        batch_size=test_batch_size, shuffle=False))
    tf.reset_default_graph()

    session_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                    log_device_placement=log_device_placement)

    with tf.Session(config=session_config) as session:
        network = TextBiLSTM(sequence_length=max_length, num_classes=num_classes, word_vocab_size=word_vocab_size,
                             word_embed_dim=embed_dim, n_hidden=n_hidden,
                             max_char_per_word=max_char_per_word, char_vocab_size=char_vocab_size,
                             char_embed_dim=char_embed_dim, grad_clip=grad_clip,
                             num_filters=num_filters, filter_size=filter_size)
        # 迭代次数
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(checkpoint_dir, MODEL_NAME))
        accuracy, accuracy_low_classes = af.test_step(
            session, network, PadZeroBegin, max_length, test_batches, dropout_keep_prob, label_alphabet,
            embed_table, char_embed_table)
        print("\t识别率 {:g}\t实体识别率 {:}".format(accuracy, accuracy_low_classes))
