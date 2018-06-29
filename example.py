# -*- coding: utf8 -*-

import numpy as np
from embeddingmachine.models.autoencoder import DNNAutoEncoder

autoencoder= DNNAutoEncoder(embedding_size=100, input_width=100, input_height=100, input_channel=3, encoder_sizes=[512, 256])
model = autoencoder.build()
model.compile(loss='mse', optimizer='sgd')

data = np.random.random((256, 100, 100, 3))
model.fit(data, data, epochs=10, batch_size=32)
