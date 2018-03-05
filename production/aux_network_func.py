#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import tensorflow as tf

from LSTMCNNCRFNER.config.config import OUTPUT_PATH


def create_feed_dict(network, PadZeroBegin, max_length, x_batch, y_batch, act_seq_lengths, dropout_keep_prob,
                     embed_table, char_batch, char_embed_table):
    """
    返回输入数据
    :param network:
    :param PadZeroBegin:
    :param max_length:
    :param x_batch:
    :param y_batch:
    :param act_seq_lengths:
    :param dropout_keep_prob:
    :param embed_table:
    :param char_batch:
    :param char_embed_table:
    :return: feed_dict
    """
    if PadZeroBegin:
        cur_batch_size = len(x_batch)
        sequence_length_batch = np.full(cur_batch_size, max_length, dtype=int)
        feed_dict = {
            network.input_x: x_batch,
            network.input_y: y_batch,
            network.dropout_keep_prob: dropout_keep_prob,
            network.word_embedding_placeholder: embed_table,
            network.sequence_lengths: sequence_length_batch,
            # NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
            # BiLSTM.sequence_lengths : seq_length
            network.input_x_char: char_batch,
            network.char_embedding_placeholder: char_embed_table

        }
    else:
        feed_dict = {
            network.input_x: x_batch,
            network.input_y: y_batch,
            network.dropout_keep_prob: dropout_keep_prob,
            network.word_embedding_placeholder: embed_table,
            # BiLSTM.sequence_lengths : sequence_length_batch, #NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
            network.sequence_lengths: act_seq_lengths,
            network.input_x_char: char_batch,
            network.char_embedding_placeholder: char_embed_table
        }
    return feed_dict


def create_feed_dict_test(network, PadZeroBegin, max_length, x_batch, y_batch, act_seq_lengths, dropout_keep_prob,
                          char_batch):
    if PadZeroBegin:
        cur_batch_size = len(x_batch)
        sequence_length_batch = np.full((cur_batch_size), max_length, dtype=int)
        feed_dict = {
            network.input_x: x_batch,
            network.input_y: y_batch,
            network.dropout_keep_prob: dropout_keep_prob,
            network.sequence_lengths: sequence_length_batch,
            # NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
            # BiLSTM.sequence_lengths : seq_length
            network.input_x_char: char_batch,

        }
    else:
        feed_dict = {
            network.input_x: x_batch,
            network.input_y: y_batch,
            network.dropout_keep_prob: dropout_keep_prob,
            # BiLSTM.sequence_lengths : sequence_length_batch, #NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
            network.sequence_lengths: act_seq_lengths,
            network.input_x_char: char_batch,

        }
    return feed_dict


def predict_accuracy_and_write(logits, transition_params, seq_length, y_batch, x_batch, label_alphabet,
                               begin_zero=True):
    correct_labels = 0
    total_labels = 0
    correct_labels_low_classes = 0
    total_labels_low_classes = 0

    for logit, y_, sequence_length_, x_ in zip(logits, y_batch, seq_length, x_batch):
        # 移除增加的额外字符 移除[:length]
        logit = logit[-sequence_length_:] if begin_zero else logit[:sequence_length_]
        y_ = y_[-sequence_length_:] if begin_zero else y_[:sequence_length_]
        x_ = x_[-sequence_length_:] if begin_zero else x_[:sequence_length_]
        # crf预测序列x_的标签值
        predict_sequence, predict_score = tf.contrib.crf.viterbi_decode(logit, transition_params)

        # y_实际值  预测值
        for xi, yi, vi in zip(x_, y_, predict_sequence):
            y_label = label_alphabet.get_instance(yi)
            predict_label = label_alphabet.get_instance(vi)
            if y_label != "O":
                total_labels_low_classes = total_labels_low_classes + 1
                if y_label == predict_label:
                    correct_labels_low_classes = correct_labels_low_classes + 1
        # 计算词标签准确率.
        correct_labels += np.sum(np.equal(predict_sequence, y_))
        total_labels += sequence_length_
    # accuracy = 100.0 * correct_labels / float(total_labels)
    # accuracy_low_classes = 100.0 * correct_labels_low_classes / float(total_labels_low_classes)

    return correct_labels, total_labels, correct_labels_low_classes, total_labels_low_classes


def test_step(session, network, PadZeroBegin, max_length, test_batches,
              dropout_keep_prob, label_alphabet, embed_table, char_embed_table):
    correct_labels = 0
    total_labels = 0
    correct_labels_low_classes = 0
    total_labels_low_classes = 0
    sentence = ['word\tlabel\tpredict']
    for test_batch in test_batches:
        x_batch, y_batch, seq_length, char_batch, word_batch = zip(*test_batch)
        feed_dict = create_feed_dict(network, PadZeroBegin, max_length, x_batch, y_batch, seq_length,
                                     dropout_keep_prob, embed_table, char_batch, char_embed_table)

        logits, transition_params, embedded_char, embedded_words, char_pool_flat, input_x_test = session.run(
            [network.logits, network.transition_params, network.W_char, network.W_word, network.char_pool_flat,
             network.input_x], feed_dict=feed_dict)
        # crf预测序列x_的标签值 移除增加的额外字符 移除[:length]
        for log_, y_, length_, x_ in zip(logits, y_batch, seq_length, word_batch):

            log_ = log_[-length_:] if PadZeroBegin else log_[:length_]
            y_ = y_[-length_:] if PadZeroBegin else y_[:length_]
            x_ = x_[-length_:] if PadZeroBegin else x_[:length_]

            predict_sequence, score = tf.contrib.crf.viterbi_decode(log_, transition_params)
            temp = []
            for xi, yi, vi in zip(x_, y_, predict_sequence):
                y_label = label_alphabet.get_instance(yi)
                predict_label = label_alphabet.get_instance(vi)
                try:

                    temp.append('\t'.join([xi, y_label, predict_label]))
                except:
                    print([xi, y_label, predict_label], [xi, yi, vi])
                if y_label != "O":
                    total_labels_low_classes = total_labels_low_classes + 1
                    if y_label == predict_label:
                        correct_labels_low_classes = correct_labels_low_classes + 1
            sentence.append('\n'.join(temp) + '\n')
            # 计算词标签准确率.
            correct_labels += np.sum(np.equal(predict_sequence, y_))
            total_labels += length_
    accuracy = 100.0 * correct_labels / float(total_labels)
    accuracy_low_classes = 100.0 * correct_labels_low_classes / float(total_labels_low_classes)

    with open(os.path.join(OUTPUT_PATH, 'test.txt'), 'w') as fp:
        fp.write('\n'.join(sentence))
    return accuracy, accuracy_low_classes


# This function is just to understand the network for debugging purposes
def viterbi_decode(score, transition_params, targetWordIndex):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indicies.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    v_target = np.zeros_like(transition_params)
    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        if t == targetWordIndex:
            v_target = v
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    if targetWordIndex == 0:
        total = float(np.sum([i if i > 0 else 0 for i in score[0]]))
        prob = [i / total if i > 0 else 0 for i in score[0]]
    else:
        total = float(np.sum([i if i > 0 else 0 for i in v_target[viterbi[targetWordIndex]]]))
        prob = [i / total if i > 0 else 0 for i in v_target[viterbi[targetWordIndex]]]
    pickle.dump(prob, open("prob.dill", 'wb'))
    '''dill.dump(trellis,open("trellis.dill",'wb'))
    dill.dump(score,open("score.dill",'wb'))
    dill.dump(transition_params,open("transition_params.dill",'wb'))'''
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score, prob


def debug(logits, transition_params, seq_length, x_batch, word_alphabet, label_alphabet, targetWordIndexArray,
          prefix_filename="Dev", beginZero=True):
    for tf_unary_scores_, sequence_length_, x_, targetWordIndex in zip(logits, seq_length, x_batch,
                                                                       targetWordIndexArray):
        # Remove padding from the scores and tag sequence.
        tf_unary_scores_ = tf_unary_scores_[-sequence_length_:] if beginZero else tf_unary_scores_[:sequence_length_]
        x_ = x_[-sequence_length_:] if beginZero else x_[:sequence_length_]

        # Compute the highest scoring sequence.
        viterbi_sequence, viterbi_score, prob = viterbi_decode(tf_unary_scores_, transition_params, targetWordIndex)

    return
