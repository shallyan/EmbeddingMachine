# -*- coding: utf8 -*-

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from embeddingmachine.models.autoencoder import DNNAutoEncoder
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# -1 ~ 1
x_train = 2 * (x_train.astype('float32') / 255. - 0.5)       
x_test = 2 * (x_test.astype('float32') / 255. - 0.5)
# 28 x 28 x 1
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

print(type(x_train))
print(x_train.shape)

autoencoder= DNNAutoEncoder(embedding_size=10, input_width=x_train.shape[1],
                input_height=x_train.shape[2], input_channel=x_train.shape[3], 
                encoder_sizes=[512, 256])
model = autoencoder.build()
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, x_train, epochs=10, batch_size=256)

encoder = autoencoder.encoder_model
enc_ret = encoder.predict(x_test)
print(type(enc_ret))
print(enc_ret.shape)
print(enc_ret[0])

