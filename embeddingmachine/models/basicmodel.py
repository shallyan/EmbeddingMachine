# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

class BasicModel(object):
  def __init__(self, embedding_size, input_width, input_height, input_channel):
    self.name = 'BasicModel'
    self.embedding_size = embedding_size
    self.input_width = input_width
    self.input_height = input_height
    self.input_channel = input_channel

  def build(self):
    raise NotImplementedError('[{}] build function has not been implemented'.format(self.name))
