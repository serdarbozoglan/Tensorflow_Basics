import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

# Use Pandas to load dataset from csv file
import pandas as pd


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0 # in reshaping the last dimension (1) is number of channels
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# CNN --> BatchNorm --> ReLU (Common strcuture)

class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    # call method will be used as forward
    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        #print(x.shape) # we can check the shape of x each time
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

# We will create something very similar to ResNet but not exactly the same one
class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.cnn1 = CNNBlock(channels[0], 3)
        self.cnn2 = CNNBlock(channels[1], 3)
        self.cnn3 = CNNBlock(channels[2], 3)
        self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], 3, padding="same") # this will be used for skip connections

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x + self.identity_mapping(input_tensor), training=training,)
        x = self.pooling(x)
        return x

# keras.Model has more functionality than layers.Layer and in the model part we should use this one
class ResNet_Like(keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D() # Bunun yerine self.flatten = layers.Flatten() da kullanilabilir de
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x, training=training)
        x = self.classifier(x)
        return x

# model = keras.Sequential(
#     [
#         CNNBlock(32),
#         CNNBlock(64),
#         CNNBlock(128),
#         layers.Flatten(),
#         layers.Dense(10)
#     ]
# )

def model(self):
    x = keras.Input(shape=(28,28,1))
    return keras.Model(inputs=[x], outputs=self.call(x))

model = ResNet_Like(num_classes=10)
model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(X_train, y_train,batch_size=64, epochs=1, verbose=2)
print(model.model().summary()) # buradaki model.model() --> yukaridaki model function'dan geliyor, overwrite ediyoruz call function'i
model.evaluate(X_test, y_test, batch_size=64)
        