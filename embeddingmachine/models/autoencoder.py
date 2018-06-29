# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape
from keras.activations import softmax
from embeddingmachine.models.basicmodel import BasicModel
from embeddingmachine.utils.utility import show_layer_info

class DNNAutoEncoder(BasicModel):
  def __init__(self, embedding_size, input_width, input_height, input_channel, encoder_sizes):
    super(DNNAutoEncoder, self).__init__(embedding_size, input_width, input_height, input_channel)
    self.name = 'DNNAutoEncoder'
    self.encoder_sizes = encoder_sizes 
    print('[{}] init done'.format(self.name), end='\n')

  def build(self):
    input_shape = (self.input_width, self.input_height, self.input_channel)
    input_obj = Input(name='input', shape=input_shape)
    show_layer_info('Input', input_obj)

    flat_input = Flatten()(input_obj)
    show_layer_info('Flatten', flat_input)
    
    # encoder
    encoder = flat_input
    for encoder_size in self.encoder_sizes:
      encoder = Dense(encoder_size, activation='relu')(encoder)
      show_layer_info('Encoder', encoder)

    encoder = Dense(self.embedding_size)(encoder)
    show_layer_info('EncoderOutput', encoder)

    # decoder
    decoder = encoder
    for decoder_size in reversed(self.encoder_sizes):
      decoder = Dense(decoder_size, activation='relu')(decoder)
      show_layer_info('Decoder', decoder)

    from operator import mul
    output_size = reduce(mul, input_shape)
    decoder = Dense(output_size, activation='tanh')(decoder)
    show_layer_info('DecoderOutput', decoder)

    output_obj = Reshape(input_shape)(decoder)
    show_layer_info('Output', output_obj)

    self.model = Model(inputs=input_obj, outputs=output_obj)
    self.encoder_model = Model(inputs=input_obj, outputs=encoder)  
    return self.model 
