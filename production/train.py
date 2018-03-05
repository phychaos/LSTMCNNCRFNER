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


def train():
    print("读取训练集和验证集...")
    # 读取训练集
    x_word_train, y_label_train, x_train, y_train = dp.read_sequence_labeling(
        TRAIN_NER_PATH, word_alphabet, label_alphabet, word_column, label_column)
    # 读取验证集
    x_word_dev, y_label_dev_, x_dev, y_dev = dp.read_sequence_labeling(
        TEST_NER_PATH, word_alphabet, label_alphabet, word_column, label_column)

    # 训练集-标签 数据补全 补全数据集 实际数据长度
    x_train_pad, train_seq_length = utils.pad_sequence(x_train, max_length, PadZeroBegin)
    y_train_pad, _ = utils.pad_sequence(y_train, max_length, PadZeroBegin)
    # 从训练集创建字符集-索引
    char_index_train, _ = dp.generate_character_data(x_word_train, char_alphabet)

    # 验证集-标签 数据补全 补全数据集 实际数据长度
    x_dev_pad, dev_seq_length = utils.pad_sequence(x_dev, max_length, PadZeroBegin)
    y_dev_pad, _ = utils.pad_sequence(y_dev, max_length, PadZeroBegin)
    # 从验证集创建字符集-索引
    char_index_dev, _ = dp.generate_character_data(x_word_dev, char_alphabet)

    # 构建字符-id 数据集 数据长度 * 句子长度 * 词汇长度
    char_index_train_pad = dp.construct_padded_char(char_index_train, char_alphabet, max_length, max_char_per_word)
    char_index_dev_pad = dp.construct_padded_char(char_index_dev, char_alphabet, max_length, max_char_per_word)

    tf.reset_default_graph()

    session_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                    log_device_placement=log_device_placement)

    with tf.Session(config=session_config) as session:
        best_accuracy = 0
        best_overall_accuracy = 0
        best_step = 0

        network = TextBiLSTM(sequence_length=max_length, num_classes=num_classes, word_vocab_size=word_vocab_size,
                             word_embed_dim=embed_dim, n_hidden=n_hidden,
                             max_char_per_word=max_char_per_word, char_vocab_size=char_vocab_size,
                             char_embed_dim=char_embed_dim, grad_clip=grad_clip,
                             num_filters=num_filters, filter_size=filter_size)
        # 迭代次数
        global_step = tf.Variable(0, name="global_step", trainable=False)
        decay_step = int(len(x_train_pad) / batch_size)

        learn_rate = tf.train.exponential_decay(start_learn_rate, global_step, decay_step, decay_rate, staircase=True)
        # 优化器
        if Optimizer == 2:
            optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        elif Optimizer == 1:
            optimizer = tf.train.AdamOptimizer(learn_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learn_rate)
        # This is the first part of minimize()
        grads_and_vars = optimizer.compute_gradients(network.loss)
        # This is the second part of minimize()
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", network.loss)
        # acc_summary = tf.summary.scalar("accuracy", BiLSTM.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(OUTPUT_PATH, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(OUTPUT_PATH, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
        session.run(tf.global_variables_initializer())

        # 产生数据集
        batches = utils.batch_iter(
            list(zip(x_train_pad, y_train_pad, train_seq_length, char_index_train_pad)), batch_size, num_epochs)

        dev_batches = list(utils.test_batch_iter(
            list(zip(x_dev_pad, y_dev_pad, dev_seq_length, char_index_dev_pad)), batch_size=test_batch_size))
        print('开始训练...')
        for batch in batches:
            x_train_batch, y_train_batch, act_seq_lengths, char_batch = zip(*batch)
            train_step(session, network, x_train_batch, y_train_batch, act_seq_lengths, char_batch,
                       global_step, train_op, train_summary_op, train_summary_writer)

            current_step = tf.train.global_step(session, global_step)
            if current_step % evaluate_every == 0:
                print("\n\t验证集: {}".format(current_step))
                accuracy, accuracy_low_classes = dev_step(session, network, dev_batches, dev_summary_op)
                if accuracy_low_classes > best_accuracy:
                    saver.save(session, checkpoint_prefix, global_step=current_step)
                    best_accuracy = accuracy_low_classes
                    best_step = current_step
                    best_overall_accuracy = accuracy
                    print("\t迭代次数 {},  识别率 {:g} 实体识别率 {:g}\n".format(best_step, best_overall_accuracy, best_accuracy))


def train_step(session, network, x_batch, y_batch, seq_lengths, char_batch, global_step,
               train_op, train_summary_op, train_summary_writer):
    feed_dict = af.create_feed_dict(network, PadZeroBegin, max_length, x_batch, y_batch, seq_lengths,
                                    dropout_keep_prob, embed_table, char_batch, char_embed_table)

    _, step, summaries, loss, logits, transition_params = session.run(
        [train_op, global_step, train_summary_op, network.loss, network.logits, network.transition_params],
        feed_dict=feed_dict)

    if step % 50 == 0:
        print("\tstep {},\t loss {:g}".format(step, loss))
    train_summary_writer.add_summary(summaries, step)


def dev_step(session, network, dev_batches, dev_summary_op):
    correct_labels = 0
    total_labels = 0
    correct_labels_low_classes = 0
    total_labels_low_classes = 0

    for dev_batch in dev_batches:
        x_batch, y_batch, seq_lengths, char_batch = zip(*dev_batch)

        feed_dict = af.create_feed_dict(network, PadZeroBegin, max_length, x_batch, y_batch, seq_lengths,
                                        dropout_keep_prob, embed_table, char_batch, char_embed_table)
        logits, transition_params, summaries = session.run([network.logits, network.transition_params, dev_summary_op],
                                                           feed_dict=feed_dict)

        correct_y, total_y, correct_y_low_classes, total_y_low_classes = af.predict_accuracy_and_write(
            logits, transition_params, seq_lengths, y_batch, x_batch, label_alphabet, begin_zero=PadZeroBegin)

        correct_labels += correct_y
        total_labels += total_y
        correct_labels_low_classes += correct_y_low_classes
        total_labels_low_classes += total_y_low_classes
    accuracy = 100.0 * correct_labels / float(total_labels)
    accuracy_low_classes = 100.0 * correct_labels_low_classes / float(total_labels_low_classes)
    return accuracy, accuracy_low_classes
