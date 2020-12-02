import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Train shape', X_train.shape, X_test.shape)
print('y train', y_train.shape)

X_train = X_train.reshape(-1, 784).astype("float32") / 255.0  # 28x28 = 784
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)

# Sequential API (Very convinent but not very flexiable)(good for one input and one output)
# 1. method Sequential API icin list seklinde tanimlamak
# model = keras.Sequential(
#     [
#     keras.Input(shape=(28*28)),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(10) # output layer, no activation func needed
#     ]
# )

# 2.method Sequential API icin add ile tanimlamak
# model = keras.Sequential()
# model.add(keras.Input(shape=28*28))
# model.add(layers.Dense(512, activation='relu'))
# print(model.summary()) # This is a debug tool, her layerdan sonra parametreler nasil degisiyor takip etmek icin
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(10))

# Functional API (A bit more flexible, multiple input, multiple output)
inputs = keras.Input(shape=28*28)
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
x = layers.Dense(128, activation='relu', name='third_layer')(x)
outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Eger herhangi bir layer'in ciktisini almak istersek. Ornegin 2.hidden layer'in ciktisini almks istiyoruz
# model = keras.Model(inputs=inputs,
#                     outputs=[model.get_layer('second_layer').output])
# eger butun layer'larin outputunu almka istersek
# model = keras.Model(inputs=inputs,
#                 outputs=[layer.output for layer in  model.layers])

# features = model.predict(X_test)
# for feature in features:
#     print(feature.shape)
# import sys
# sys.exit()

print(model.summary()) # model summary'sini lamak icin keras input tanimlanmali ya da tanimlamadiysak mdoel.fit'ten sonra cagirabiliriz model.summary()'i
# import sys #if we want to exit the code
# sys.exit()
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False), # eger Functional API icinde output layerda activation='softmax' yaptiysan from_logits=False olmali
    # default argument is from_logits=False, tanimlamadi isen layers'ta it should be True
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    metrics=['accuracy']
)

model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=2)
model.evaluate(X_test, y_test, batch_size=32, verbose=2) # no epochs needed here





