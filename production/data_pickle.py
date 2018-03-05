#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 林利芳
# @File: data_pickle.py
# @Time: 18-2-5 下午3:41
from config.parameter import *
from core import utils
from core.utils import save_data, load_data
from production import data_processor as dp

from LSTMCNNCRFNER.config.config import *
from LSTMCNNCRFNER.production.alphabet import Alphabet


def format_data_index():
    # 构建词典-索引 标签-索引
    word_alphabet = Alphabet('word2index')
    label_alphabet = Alphabet('ner2index')
    char_alphabet = Alphabet('character')
    # 读取训练集
    word_train, _, word_index_train, label_index_train = dp.read_sequence_labeling(
        TRAIN_NER_PATH, word_alphabet, label_alphabet, word_column, label_column, out_dir=DATA_DICT_PATH)
    word_alphabet.close()

    max_length_train = utils.get_max_length(word_train)

    max_length = min(dp.MAX_LENGTH, max_length_train)

    # 从训练集创建字符集-索引
    char_index_train, max_char_per_word_train = dp.generate_character_data(word_train, char_alphabet)
    # 停止增加字符集
    char_alphabet.close()

    max_char_per_word = min(dp.MAX_CHAR_PER_WORD, max_char_per_word_train)

    # 构建字符集向量 向量大小 字符个数* 30 值大小介于-sqrt(3.0/30) -> sqrt(3.0/30)
    char_embed_table = dp.build_char_embed_table(char_alphabet, char_embed_dim=char_embed_dim)

    # 词汇 字符 标签
    word_vocab = word_alphabet.instances
    word_vocab_size = len(word_vocab)
    char_vocab = char_alphabet.instances
    char_vocab_size = len(char_vocab)
    num_classes = len(label_alphabet.instances) + 1

    # 保存参数
    flags_dict = dict()
    flags_dict['max_length'] = max_length
    flags_dict['num_classes'] = num_classes
    flags_dict['word_vocab_size'] = word_vocab_size
    flags_dict['char_vocab_size'] = char_vocab_size
    flags_dict['max_char_per_word'] = max_char_per_word
    flags_dict['char_embed_table'] = char_embed_table

    # 持久化参数
    checkpoint_dir = os.path.abspath(os.path.join(OUTPUT_PATH, "checkpoints"))
    flags_dict['checkpoint_dir'] = checkpoint_dir
    save_data(flags_dict, filename=os.path.join(DATA_DICT_PATH, "config.pkl"))
    save_data(char_alphabet, filename=os.path.join(DATA_DICT_PATH, "char_alphabet.pkl"))
    save_data(word_alphabet, filename=os.path.join(DATA_DICT_PATH, "word_alphabet.pkl"))
    save_data(label_alphabet, filename=os.path.join(DATA_DICT_PATH, "label_alphabet.pkl"))


def load_data_dict():
    """
    加载参数
    :return:
    """
    flags_dict = load_data(filename=os.path.join(DATA_DICT_PATH, "config.pkl"))
    char_alphabet = load_data(filename=os.path.join(DATA_DICT_PATH, "char_alphabet.pkl"))
    word_alphabet = load_data(filename=os.path.join(DATA_DICT_PATH, "word_alphabet.pkl"))
    label_alphabet = load_data(filename=os.path.join(DATA_DICT_PATH, "label_alphabet.pkl"))
    return flags_dict, word_alphabet, label_alphabet, char_alphabet
