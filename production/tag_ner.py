#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @File: tag_ner.py
# @Time: 18-2-6 下午2:22
import re

import jieba

from LSTMCNNCRFNER.core.utils import write_txt


def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            word, ner, keyword = line.strip().split('*&*')
            data.append([word, ner, keyword])
    return data


def tag_ner(xml_data_path, result_path):
    data = read_data(xml_data_path)
    tag_data = []
    for word, ner, keyword in data:
        temp_data = []
        if ner:
            pattern = re.compile(ner)
            last = 0
            for i in pattern.finditer(word):

                start, end = i.span()
                pre_word = word[last:start].strip().split()
                for w in pre_word:
                    temp_data.append(w + '\tO\n')

                tag_word = word[start:end].strip().split()
                key_tag = ['B'] + ['I'] * (len(tag_word) - 1)
                for w, tag in zip(tag_word, key_tag):
                    temp_data.append(w + '\t' + tag + '\n')
                last = end
            if last < len(word):
                pre_word = word[last:].strip().split()
                for w in pre_word:
                    temp_data.append(w + '\tO\n')
        else:
            tag_word = word.strip().split()
            for w in tag_word:
                temp_data.append(w + '\tO\n')

        tag_data.append(''.join(temp_data))
    # data_size = len(tag_data)
    # indices = np.random.permutation(np.arange(data_size))
    # print(indices)
    # shuffled_data = [tag_data[i] for i in indices]
    # shuffled_data = tag_data

    # index = int(data_size * 0.8)
    # train_data = shuffled_data[:index]
    # test_data = shuffled_data[index:]
    write_txt(result_path, tag_data)
    # write_txt(TEST_NER_PATH, test_data)


def add_tag(word, tag):
    data = []
    for w, t in zip(word, tag):
        data.append('\t'.join([w, t]))
    return '\n'.join(data) + '\n'


def cut_word(data):
    words = jieba.cut(data)
    return words
