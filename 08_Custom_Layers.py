import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

# Use Pandas to load dataset from csv file
import pandas as pd

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0 
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def build(self, input_shape):
        self.w = self.add_weight(
            name = "w", # !!!!!! if we do not specify a name, we cannot save a model
            shape = (input_shape[-1], self.units), # input_shape'in son dimensioni 784 cunku yukaridaki reshape yaptik
            initializer = "random_normal",
            trainable = True,
        )
        self.b = self.add_weight(
            name = 'b',
            shape =(self.units, ),
            initializer = 'zeros',
            trainable = True, 
        )

class MyReLU(layers.Layer):
    def __init__(self):
        super(MyReLU, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0) # we are defining ReLU in this way here

class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64) # input shape'i deklare etmemize gerek yok, build in icinde tanimladik inpit_shape[-1] olarak
        self.dense2 = Dense(num_classes) # input shape'i deklare etmemize gerek yok
        self.relu = MyReLU()

    def call(self, input_tensor):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)


model = MyModel()
model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
)

model.fit(X_train, y_train, batch_size=64, epochs=20)
model.evaluate(X_test, y_test, batch_size=64)





