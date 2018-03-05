#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle

import core.utils as utils
import numpy as np

from LSTMCNNCRFNER.config.parameter import MAX_LENGTH, MAX_CHAR_PER_WORD, word_end

logger = utils.get_logger("LoadData")


def read_sequence_labeling(filename, word_alphabet, label_alphabet, word_column=1, label_column=3, out_dir=None):
    """
    读取数据 转化为词 标签 词索引 标签索引
    :param filename: 文件路径
    :param word_column: 词的列 第0列
    :param label_column: 标签的列 第3列
    :param word_alphabet: 词典-索引
    :param label_alphabet: 标签-索引
    :param out_dir: 输出路径
    :return: 句子的词和标签 以及相应的索引.
    """
    word_sentences = []
    label_sentences = []
    word_index_sentences = []
    label_index_sentences = []
    words = []
    labels = []
    word_ids = []
    label_ids = []
    vocab = set()
    num_tokens = 0
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            tokens = line.strip().split()
            # 添加词及标签

            if len(tokens) > label_column:
                word = tokens[word_column]
                label = tokens[label_column]
                words.append(word)
                labels.append(label)

                vocab.add(word)

                word_id = word_alphabet.get_index(word)
                label_id = label_alphabet.get_index(label)
                word_ids.append(word_id)
                label_ids.append(label_id)
            # 添加句子 line=''
            else:
                if 0 < len(words) <= MAX_LENGTH:
                    word_sentences.append(words)
                    label_sentences.append(labels)
                    word_index_sentences.append(word_ids)
                    label_index_sentences.append(label_ids)
                    num_tokens += len(words)
                words = []
                labels = []
                word_ids = []
                label_ids = []
    if 0 < len(words) <= MAX_LENGTH:
        word_sentences.append(words)
        label_sentences.append(labels)

        word_index_sentences.append(word_ids)
        label_index_sentences.append(label_ids)
        num_tokens += len(words)

    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        vocab_save_path = os.path.join(out_dir, "vocab.pkl")
        with open(vocab_save_path, 'wb') as fp:
            pickle.dump(vocab, fp)
    return word_sentences, label_sentences, word_index_sentences, label_index_sentences


def build_embed_table(word_alphabet, embed_dict, embed_dim, case_less):
    """
    为词汇构建词向量, 词向量来源于glove
    :param word_alphabet: 词典-索引
    :param embed_dict: glove词向量
    :param embed_dim: 词向量维度
    :param case_less: True为glove False为word2vec
    :return:
    """
    scale = np.sqrt(3.0 / embed_dim)
    # 词典大小[词+UNK] 向量维度
    embed_table = np.empty([word_alphabet.size(), embed_dim], dtype=np.float64)
    embed_table[word_alphabet.default_index, :] = np.random.uniform(-scale, scale, [1, embed_dim])
    for word, index in word_alphabet.items():
        ww = word.lower() if case_less else word
        embed = embed_dict[ww] if ww in embed_dict else np.random.uniform(-scale, scale, [1, embed_dim])
        embed_table[index, :] = embed
    return embed_table


def construct_padded_char(index_sentences, char_alphabet, max_sent_length, max_char_per_word):
    """
    构建字符id 维度 数据长度 句子长度 词汇长度
    :param index_sentences:字符数据集
    :param char_alphabet:字符-标签
    :param max_sent_length:最大句子长度
    :param max_char_per_word:最大词汇长度
    :return: 三维数组 数据长度 句子长度 词汇长度
    """
    # 数据集大小 最大句子长度 最大词长度
    data_char_index = np.empty([len(index_sentences), max_sent_length, max_char_per_word], dtype=np.int32)
    # 词尾id=0
    word_end_id = char_alphabet.get_index(word_end)

    for i in range(len(index_sentences)):
        words = index_sentences[i]
        sent_length = len(words)
        for j in range(min(sent_length, max_sent_length)):
            chars = words[j]
            char_length = len(chars)
            for k in range(min(char_length, max_char_per_word)):
                cid = chars[k]
                data_char_index[i, j, k] = cid
            # 词尾补充为word_end_id
            data_char_index[i, j, char_length:] = word_end_id
        # 句子长度大于sent_length补0
        data_char_index[i, sent_length:, :] = 0
    return data_char_index


def build_char_embed_table(char_alphabet, char_embed_dim=30):
    """
    字符嵌入
    :param char_alphabet:
    :param char_embed_dim:词嵌入维度
    :return:
    """
    scale = np.sqrt(3.0 / char_embed_dim)
    # 构建随机数组 长度26*30 值 -0.31-0.31
    char_embed_table = np.random.uniform(-scale, scale, [char_alphabet.size(), char_embed_dim]).astype(np.float64)
    return char_embed_table


def generate_character_data(sentences_list, char_alphabet):
    """
    将数据集转化为字符集ID 返回字符id及最大词长度
    :param sentences_list: 数据集
    :param char_alphabet: 字符-索引
    :return: 句子字符索引 最大词长度
    """
    char_alphabet.get_index(word_end)
    index_sentences = []
    max_length = 0
    for words in sentences_list:
        index_words = []
        for word in words:
            index_chars = []
            if len(word) > max_length:
                max_length = len(word)

            for char in word[:MAX_CHAR_PER_WORD]:
                char_id = char_alphabet.get_index(char)
                index_chars.append(char_id)

            index_words.append(index_chars)
        index_sentences.append(index_words)

    max_char_per_word = min(MAX_CHAR_PER_WORD, max_length)
    return index_sentences, max_char_per_word
