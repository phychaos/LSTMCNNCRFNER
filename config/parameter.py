#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @File: parameter.py
# @Time: 18-2-5 下午3:33

dropout_keep_prob = 0.5
grad_clip = 5
max_global_clip = 5.0
batch_size = 64
test_batch_size = 64
word_column = 0
label_column = 2
n_hidden = 200
num_epochs = 50
num_filters = 30
filter_size = 3
evaluate_every = 100  # 评估次数

char_embed_dim = 30  # 字符维度

Optimizer = 1  # Adam : 1 , SGD:2 优化器
num_checkpoints = 5  # 模型数量

start_learn_rate = 0.015  # 起始学习率
decay_rate = 0.05
allow_soft_placement = True
log_device_placement = False
PadZeroBegin = False

MAX_LENGTH = 120
MAX_CHAR_PER_WORD = 45
root_symbol = "##ROOT##"
root_label = "<ROOT>"
word_end = "##WE##"

MODEL_NAME = 'model-1800'
