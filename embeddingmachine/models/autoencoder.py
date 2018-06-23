# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.activations import softmax
from model import BasicModel
from utils.utility import show_layer_info

class DNNAutoEncoder(BasicModel):
  def __init__(self, config):
    super(AutoEncoder, self).__init__(config)
    self.name = 'DNNAutoEncoder'
    self.check_list = ['embedding_size', 'encoder_sizes', 'input_width', 'input_height', 'input_channel']
    self.setup()
    self.check()
    print('[{}] init done'.format(self.name), end='\n')

  def setup(self):
    self.config.setdefault('encoder_sizes', [512, 256])

  def build(self):
    input_shape = (self.config['input_width'], self.config['input_height'], self.config['input_channel'])
    input_obj = Input(name='input', shape=input_shape)
    show_layer_info('Input', input_obj)

    flat_input = Flatten()(input_obj)
    show_layer_info('Flatten', flat_input)
    
    # encoder
    encoder = flat_input
    for encoder_size in self.config['encoder_sizes']:
      encoder = Dense(encoder_size, activation='relu')(encoder)
      show_layer_info('Encoder', encoder)

    encoder = Dense(self.config['embedding_size'])(encoder)
    show_layer_info('EncoderOutput', encoder)

    # decoder
    decoder = encoder
    for decoder_size in reversed(self.config['encoder_sizes']):
      decoder = Dense(decoder_size, activation='relu')(decoder)
      show_layer_info('Decoder', decoder)

    from operator import mul
    output_size = reduce(mul, input_shape)
    decoder = Dense(output_size)(decoder)
    show_layer_info('DecoderOutput', decoder)

    output_obj = Reshape(input_shape)(decoder)
    show_layer_info('Output', output_obj)

    self.model = Model(inputs=input_obj, outputs=output_obj)
    self.encoder_model = Model(inputs=input_obj, outputs=encoder)  
    return self.model 
