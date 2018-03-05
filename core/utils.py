#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @File: utils.py
# @Time: 2018/2/4 21:34

import csv
import logging
import pickle
import sys

import numpy as np
import xlrd
import xlwt
from gensim.models.word2vec import Word2Vec

from LSTMCNNCRFNER.config.config import RAW_DATA_PATH, TRAIN_NER_PATH, TEST_NER_PATH


def save_data(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


def load_data(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data


def read_xls(filename, sheet_name=None):
    """
    读取excel数据
    :param filename:文件名称
    :param sheet_name:文件名称
    :return: 列表
    """
    file = xlrd.open_workbook(filename)
    if sheet_name:
        sheet = file.sheet_by_name(sheet_name)
    else:
        sheet = file.sheet_by_index(0)
    rows = sheet.nrows
    result_list = []
    for row in range(rows):
        result_list.append(sheet.row_values(row))
    return result_list


def write_index_xls(filename, data, sheet_name='sheet1'):
    """
    写入excel
    :param filename:
    :param data:
    :param sheet_name:
    :return:
    """
    fp = xlwt.Workbook(encoding='utf8')
    sheet = fp.add_sheet(sheet_name)

    for raw, line in enumerate(data):
        sheet.write(raw, 0, raw + 1)
        for col, w in enumerate(line):
            sheet.write(raw, col + 1, w)

    fp.save(filename)


def write_xls(filename, data, sheet_name='sheet1'):
    """
    写入excel
    :param filename:
    :param data:
    :param sheet_name:
    :return:
    """
    fp = xlwt.Workbook(encoding='utf8')
    sheet = fp.add_sheet(sheet_name)

    for raw, line in enumerate(data):
        for col, w in enumerate(line):
            sheet.write(raw, col, w)

    fp.save(filename)


def write_title_xls(filename, title, data, sheet_name='sheet1'):
    """
    写入excel
    :param filename:
    :param title:
    :param data:
    :param sheet_name:
    :return:
    """
    fp = xlwt.Workbook(encoding='utf8')
    sheet = fp.add_sheet(sheet_name)

    for i, w in enumerate(title):
        sheet.write(0, i, w)

    for raw, line in enumerate(data):
        for col, w in enumerate(line):
            sheet.write(raw + 1, col, w)
    fp.save(filename)


def write_txt(filename, data):
    with open(filename, 'w', encoding='utf8')as fp:
        fp.write('\n'.join(data))


def read_txt(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        data = fp.read()
    return data


def write_csv(filename, title, data, delimiter=','):
    """
    写入csv
    :param filename:
    :param title:
    :param data:
    :param delimiter:
    :return:
    """
    with open(filename, 'w', encoding='utf8', newline="") as fp:
        csv_write = csv.DictWriter(fp, delimiter=delimiter, fieldnames=title)
        csv_write.writerows(data)


def read_csv(filename):
    """
    csv文件读取
    :param filename:文件名称
    :return:
    """
    result = []
    with open(filename, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            if len(row) >= 4:
                result.append(row[:4])
    return result


def format_csv():
    data = read_csv(RAW_DATA_PATH)
    temp = []
    result = []
    print(data[-20:])
    for row in data[1:]:
        sentence, word, pos, tag = row
        word_tag = '\t'.join([word, pos, tag])
        if word == '0' and pos == '0' and tag == 'O':
            continue
        if 'Sentence' in sentence and temp:
            result.append('\n'.join(temp) + '\n')
            temp = [word_tag]
        elif 'Sentence' in sentence and temp is []:
            temp = [word_tag]
        else:
            temp.append(word_tag)
    if temp:
        result.append('\n'.join(temp))
    print(len(result))
    index = int(len(result) * 0.9)
    train_ner = result[:index]
    test_ner = result[index:]
    write_txt(TRAIN_NER_PATH, train_ner)
    write_txt(TEST_NER_PATH, test_ner)


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """
    日志信息
    :param name: 日志名称
    :param level: 日志级别 info warning debug error
    :param handler:
    :param formatter: 格式
    :return: logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def print_flags(flags):
    """
    参数打印 写入日志
    :param flags:
    :return:
    """
    flags_dict = {}
    for attr, value in sorted(flags.__flags.items()):
        flags_dict[attr] = value
    return flags_dict


def load_word_embedding_dict(embedding, embedding_path):
    """
    从文件中读取词向量
    :param embedding: 词嵌入类型
    :param embedding_path: 词嵌入路径
    :return: 词向量字典 词向量维度
    """
    if embedding == 'word2vec':
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embed_dim = word2vec.vector_size
        return word2vec, embed_dim, False
    elif embedding == 'glove':
        embed_dim = -1
        embed_dict = dict()
        with open(embedding_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embed_dim < 0:
                    embed_dim = len(tokens) - 1
                else:
                    assert (embed_dim + 1 == len(tokens))
                embed = np.empty([1, embed_dim], dtype=np.float64)
                embed[:] = tokens[1:]
                embed_dict[tokens[0]] = embed
        return embed_dict, embed_dim, True
    else:
        raise ValueError("词嵌入需从 [word2vec, glove] 选取")


def get_max_length(word_sentences):
    """
    获取数据集句子的最大长度
    :param word_sentences: 数据集
    :return: 最大长度
    """
    max_len = 0
    for sentence in word_sentences:
        length = len(sentence)
        if length > max_len:
            max_len = length
    return max_len


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    数据batch
    :param data: 数据集
    :param batch_size:batch大小
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # 洗牌 重新排序数据
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def test_batch_iter(data, batch_size, shuffle=True):
    """
    数据batch
    :param data: 数据集
    :param batch_size:batch大小
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # 洗牌 重新排序数据
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def pad_sequence(data_set, max_length, begin_zero=True):
    """
    为数据集补全缺失值 begin_zero =True 前补， begin_zero 后补
    :param data_set:数据集
    :param max_length:数据最大长度
    :param begin_zero:前补或后补
    :return:补全数据集, 实际序列长度
    """
    # 补全数据集  实际序列长度
    data_set_p = []
    actual_sequence_length = []
    for x in data_set:
        row_length = len(x)
        if row_length <= max_length:
            actual_sequence_length.append(row_length)
            # 前补 或后补
            if begin_zero:
                data_set_p.append(np.pad(x, pad_width=(max_length - row_length, 0), mode='constant', constant_values=0))
            else:
                data_set_p.append(np.pad(x, pad_width=(0, max_length - row_length), mode='constant', constant_values=0))
        # 截断数据
        else:
            actual_sequence_length.append(max_length)
            data_set_p.append(x[0:max_length])
    return np.array(data_set_p), actual_sequence_length
