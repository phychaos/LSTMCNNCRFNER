#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle


class Alphabet:
    def __init__(self, name='word2index.pkl', keep_growing=True):
        """
        初始化
        :param name: 词典名称
        :param keep_growing:是否增加词典
        """
        self.__name = name

        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.default_index = 0  # 默认索引0 为通配符
        self.next_index = 1  # 词典从1开始

    def add(self, instance):
        """
        新增实例 并添加索引
        :param instance:
        :return:
        """
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        """
        获取索引
        :param instance:
        :return:
        """
        if self.instance2index.get(instance):
            return self.instance2index[instance]
        elif self.keep_growing:
            index = self.next_index
            self.add(instance)
            return index
        else:
            return self.default_index

    def get_instance(self, index):
        """
        索引从1开始,  第一个元素是通配符.
        :param index:
        :return:
        """
        if index == 0:
            return None
        elif 0 < index <= len(self.instances):
            return self.instances[index - 1]
        else:
            return self.instances[0]

    def size(self):
        """
        词大小
        :return:
        """
        return len(self.instances) + 1

    def items(self):
        """
        返回词典 索引列表
        :return:
        """
        return self.instance2index.items()

    def enumerate_items(self, start=1):
        """
        返回从start开始的词及索引
        :param start: 开始节点
        :return:
        """
        if start < 1 or start >= self.size():
            raise IndexError("枚举只允许位于 1 : size 区间")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        """
        停止增加词典
        :return:
        """
        self.keep_growing = False

    def open(self):
        """
        继续增加词典
        :return:
        """
        self.keep_growing = True

    def get_content(self):
        """
        获取词典内容
        :return:
        """
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        """
        从json中读取词典-索引
        :param data:
        :return:
        """
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        保存词典-索引 及词典列表
        :param output_directory: 文件路径
        :param name: 文件名称
        :return:
        """
        saving_name = name if name else self.__name
        with open(os.path.join(output_directory, saving_name), 'wb') as fp:
            pickle.dump(self.get_content(), fp)
            # try:
            #     json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
            # except Exception as e:
            #     self.logger.warn("Alphabet is not saved: %s" % repr(e))

    def load(self, input_directory, name=None):
        """
        加载词典-索引 词典列表
        :param input_directory:文件路径
        :param name: 文件名称
        :return:
        """
        loading_name = name if name else self.__name
        with open(os.path.join(input_directory, loading_name), 'rb') as fp:
            self.from_json(pickle.load(fp))
