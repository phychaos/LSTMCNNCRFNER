#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 林利芳
# @File: load_mnist.py
# @Time: 18-2-8 下午3:09
import numpy as np
import struct
import gzip
import pickle
import numpy as np


def load_data(filename):
    print(filename)
    with open(filename, 'rb', encoding='iso-8859-1') as fp:
        data = pickle.load(fp)
    x_train, y_train = data[0]
    x_val, y_val = data[1]
    x_test, y_test = data[2]
    print(x_train.shape)
    print(y_train.shape)
    x_train = x_train.reshape((-1, 1, 28, 28))
    x_val = x_val.reshape((-1, 1, 28, 28))
    x_test = x_test.reshape((-1, 1, 28, 28))
    print(x_train.shape)
    print(y_train.shape)
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_data_mnist(filename):
    with np.load(filename) as fp:
        x_train = fp['x_train']
        y_train = fp['y_train']
        x_test = fp['x_test']
        y_test = fp['y_test']
    return (x_train, y_train), (x_test, y_test)


def load_image(filename):
    with open(filename, 'rb') as fp:
        buffers = fp.read()
    head = struct.unpack_from('>IIII', buffers, 0)

    offset = struct.calcsize('>IIII')
    img_num = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = img_num * width * height
    print(bits, img_num, width, height)
    bits_string = '>' + str(bits) + 'B'  # like '>47040000B'

    images = struct.unpack_from(bits_string, buffers, offset)

    images = np.reshape(images, [img_num, width, height])
    return images


def load_label(filename):
    with open(filename, 'rb') as fp:
        buffers = fp.read()
    head = struct.unpack_from('>II', buffers, 0)
    img_num = head[1]

    offset = struct.calcsize('>II')
    num_string = '>' + str(img_num) + "B"
    labels = struct.unpack_from(num_string, buffers, offset)
    labels = np.reshape(labels, [img_num, 1])
    return labels
