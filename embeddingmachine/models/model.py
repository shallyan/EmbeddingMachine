# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

class BasicModel(object):
  def __init__(self, config):
    self.name = 'BasicModel'

    if not isinstance(config, dict):
      raise TypeError('[{}] config should be dict:'.format(self.name), config)
    self.config = config

    self.check_list = []

  def check(self):
    for e in self.check_list:
      if e not in self.config:
        raise KeyError('[{}] {} parameter not in config'.format(self.name, e))

  def build(self):
    raise NotImplementedError('[{}] build function has not been implemented'.format(self.name))
