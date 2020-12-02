import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, Sequential, regularizers
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = Sequential()
model.add(keras.Input(shape=(None, 28))) # Burada image'in her bir time step'te image'in bir satirini input olarak alacagiz.
# Image 28 by 28 oldugu icin her bir time step'te 28 pixel' (satir) i input olarak alacagiz ve kac tane satir oldugunu onceden 
# belirtmek zorunda degiliz, bu None bilinmeyen satir sayisini temsil ediyor
# Yani burada 28 time step var anlamina gelmektedir
model.add(layers.SimpleRNN(256, return_sequences=True, activation='relu')) # 512 nodes, Bunda sonra RNN layer gelecekse return_sequence=True olmali
model.add(layers.SimpleRNN(256, activation='relu'))

## GRU yapmak icin
# model.add(layers.GRU(256, return_sequences=True, activation='relu')) # 512 nodes, Bunda sonra RNN layer gelecekse return_sequence=True olmali
# model.add(layers.GRU(256, activation='relu'))

## LSTM yapmak icin
# model.add(layers.LSTM(256, return_sequences=True, activation='relu')) # 512 nodes, Bunda sonra RNN layer gelecekse return_sequence=True olmali
# model.add(layers.LSTM(256, activation='relu'))

## Bidirectional LSTM icin
# model.add(layers.Bidirectional(layers.GRU(256, return_sequences=True, activation='relu'))) # 512 nodes, Bunda sonra RNN layer gelecekse return_sequence=True olmali
# model.add(layers.Bidirectional(layers.GRU(256, activation='relu')))

model.add(layers.Dense(10)) # output nodes

print(model.summary())

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # softmax i kullanmadik so from_logits i unutma
            optimizer=keras.optimizers.Adam(lr=0.001) ,
            metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=20)
model.evaluate(X_test, y_test, batch_size=64)

