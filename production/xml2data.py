#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
# @File: xml2data.py
# @Time: 18-2-6 下午1:33
from xml.parsers.expat import ParserCreate

from LSTMCNNCRFNER.core.utils import read_txt, write_txt

pattern = re.compile('\s+')


class HandlerXML(object):
    def __init__(self):
        self.data = []
        self.tag_data = {}
        self.name = ''
        self.text = ''

    def start_element(self, name, attrs):
        self.name = name
        self.text = ''

    def end_element(self, name):
        if name == 'clause':
            self.data.append(self.tag_data)
            self.tag_data = {}
        elif self.name == 'text':
            self.tag_data['text'] = pattern.sub(' ', self.text.strip())
        elif self.name == 'cause':
            self.tag_data['ner'] = pattern.sub(' ', self.text.strip())
        elif self.name == 'keywords':
            self.tag_data['key'] = pattern.sub(' ', self.text.strip())
        self.name = ''

    def char_data(self, text):
        self.text += text.strip()


def deal_xml(xml_path, xml_data_path):
    handler = HandlerXML()

    parser = ParserCreate()

    parser.StartElementHandler = handler.start_element
    parser.EndElementHandler = handler.end_element
    parser.CharacterDataHandler = handler.char_data
    xml = read_txt(xml_path)
    parser.Parse(xml)
    data = []
    for sub_data in handler.data:
        if sub_data.get('text'):
            temp = [sub_data['text'], sub_data.get('ner', ''), sub_data.get('key', '')]
            data.append('*&*'.join(temp))

    write_txt(xml_data_path, data)
