import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, Sequential, regularizers
from tensorflow.keras.datasets import cifar10


# IF WE ARE USING CNN WE SHOULD NOT RESHAPE THE DATA
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype('float32') / 255.0

# Sequential
# model = Sequential()
# model.add(keras.Input(shape=(32, 32, 3))) # 3 color channel
# model.add(layers.Conv2D(32, 3,  padding='valid', activation='relu')) # the output channel we want is 32, kernel size=3, it will be expended to 3x3 
# # valid is the default padding which the output will be changed based on the kernel size, if we use "same" padding the  output size will remain the same
# model.add(layers.MaxPooling2D(pool_size=(2,2))) # MaxPool 2x2 yaptigimizda output input'un yarisi olur
# model.add(layers.Conv2D(64, 3, activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(128, 3, activation='relu', ))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
# print(model.summary())

def my_model():
    inputs = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x) # if we use BatchNorm activation func must be used after that
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x) # default pool size is 2 by 2
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = my_model()

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=['accuracy'])

model.fit(X_train, y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.1)

model.evaluate(X_test, y_test, batch_size=64)