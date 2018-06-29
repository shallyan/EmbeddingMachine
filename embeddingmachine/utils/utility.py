# -*- coding=utf-8 -*-

import sys
import resource

def show_layer_info(layer_name, layer_out):
  print('[Layer]: {}\t[Shape]: {} \n{}'.format(
        layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))

def show_memory_use():
  rusage_denom = 1024.
  if sys.platform == 'darwin':
    rusage_denom = rusage_denom * rusage_denom
  ru = resource.getrusage(resource.RUSAGE_SELF)
  total_memory = 1. * (ru.ru_maxrss + ru.ru_ixrss + ru.ru_idrss + ru.ru_isrss) / rusage_denom
  strinfo = "\x1b[33m [Memory] Total Memory Use: {:.4f} MB \t Resident: {} Shared: {} UnshareData: {} UnshareStack: {} \x1b[0m".format( 
            total_memory, ru.ru_maxrss, ru.ru_ixrss, ru.ru_idrss, ru.ru_isrss)
  return strinfo

def import_class(import_str):
  mod_str, _sep, class_str = import_str.rpartition('.')
  __import__(mod_str)
  try:
    return getattr(sys.modules[mod_str], class_str)
  except AttributeError:
    raise ImportError('Class {} cannot be found ({})'.format(
                      class_str, traceback.format_exception(*sys.exc_info())))

def import_object(import_str, *args, **kwargs):
  return import_class(import_str)(*args, **kwargs)
