#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @File: config.py
# @Time: 2018/2/4 21:22

import os

PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'data')
OUTPUT_PATH = os.path.join(PATH, 'output')

RAW_DATA_PATH = os.path.join(DATA_PATH, 'ner_dataset_utf8.csv')

# 训练集-测试集
TRAIN_NER_PATH = os.path.join(DATA_PATH, 'train.txt')
TEST_NER_PATH = os.path.join(DATA_PATH, 'test.txt')

GROVE_PATH = os.path.join(DATA_PATH, 'glove.6B.100d.txt')

DATA_DICT_PATH = os.path.join(PATH, 'format_data')

embedding = "glove"
embedding_path = GROVE_PATH

XML_TRAIN_PATH = os.path.join(PATH, 'data/emotion_cause_english_train.xml')
XML_TEST_PATH = os.path.join(PATH, 'data/emotion_cause_english_test.xml')
XML_TRAIN_DATA_PATH = os.path.join(PATH, 'data/emotion_en_train.txt')
XML_TEST_DATA_PATH = os.path.join(PATH, 'data/emotion_en_test.txt')

MNIST_TRAIN_IMAGES = os.path.join(DATA_PATH, 'train-images-idx3-ubyte.gz')
MNIST_TRAIN_LABELS = os.path.join(DATA_PATH, 'train-labels-idx1-ubyte.gz')
MNIST_TEST_IMAGES = os.path.join(DATA_PATH, 't10k-images-idx3-ubyte.gz')
MNIST_TEST_LABELS = os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte.gz')

MNIST_DATA = os.path.join(DATA_PATH, 'mnist.npz')
