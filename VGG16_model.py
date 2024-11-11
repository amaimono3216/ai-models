import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers
from keras.datasets import cifar10
import cv2
import numpy as np
 
 
 #データ読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes=10
#データリサイズ
img_rows=224
img_cols=224

x_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_train[::100,:,:,:]])
x_test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_test[::100,:,:,:]])

# データ正規化
x_train=x_train.astype('float32')
x_train/=255
x_test=x_test.astype('float32')
x_test/=255

# one-hotベクトル化
y_train = y_train[::100]
y_test = y_test[::100]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Convnetsへの入力は固定サイズの224*224RGB画像 
input_shape=(224, 224, 3)

#畳み込み層：flter(3*3)、ストライド(1ピクセル固定)、パディング(固定)
#最大プーリング：ストライド(2)、サイズ(2*2)
#kerasで実装
model = models.Sequential([
    #第一層
    keras.layers.Conv2D(64, (3,3), strides=(1,1), padding="same", activation="relu", input_shape=input_shape),
    keras.layers.Conv2D(64, (3,3), strides=(1,1),padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
    
    #第二層
    keras.layers.Conv2D((128), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D((128), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
    
    #第三層
    keras.layers.Conv2D((256), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D((256), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D((256), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
    
    #第四層
    keras.layers.Conv2D((512), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D((512), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D((512), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
    
    #第五層
    keras.layers.Conv2D((512), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D((512), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.Conv2D((512), (3,3), strides=(1,1), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
    
    #全結合層
    keras.layers.Flatten(),
    keras.layers.Dense(units=4096, activation="relu"),
    #keras.layers.Droupout(0.5),
    keras.layers.Dense(units=4096, activation="relu"),
    #keras.layers.Droupout(0.5),
    
    #出力層
    keras.layers.Dense(units=num_classes, activation="softmax")
])

model.summary()