import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers
from keras.datasets import cifar10
import cv2
import numpy as np

#残差ユニット
#入力をスキップ接続を通して出力に直接追加
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        #メインパスの層を定義
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        #スキップ接続の層を定義
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
            
    def call(self, inputs):
        #メインパスを通して計算
        Z = inputs 
        for layer in self.main_layers:
            Z = layer(Z)
        #スキップ接続を通して計算
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        #スキップ接続とメインパスの出力を足して、活性化関数を適用
        return self.activation(Z + skip_Z)
#ResNet34の構築   
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding="same", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
])
#ResNetの各残差ユニットを追加
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
    
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())           
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()