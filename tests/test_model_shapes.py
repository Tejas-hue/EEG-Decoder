import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from train_eegnet import build_eegnet


def test_eegnet_output_shape():
    num_classes = 2
    C, T = 22, 256
    model = build_eegnet(num_classes=num_classes, num_channels=C, num_samples=T)
    x = np.random.randn(4, C, T, 1).astype(np.float32)
    y = model(x)
    assert y.shape == (4, num_classes)


def test_eegnet_train_step():
    num_classes = 2
    C, T = 22, 256
    model = build_eegnet(num_classes=num_classes, num_channels=C, num_samples=T)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    x = np.random.randn(8, C, T, 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(8,)).astype(np.int64)
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
    assert history.history["loss"][0] >= 0.0