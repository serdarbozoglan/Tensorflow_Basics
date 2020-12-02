import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

# Use Pandas to load dataset from csv file
import pandas as pd

# HYPERPARAMETERS
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001

# Make sure we don't get any GPU errors
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_df = pd.read_csv("/Users/serdar/DATASETS/2_digit_Mnist/train.csv")
test_df = pd.read_csv("/Users/serdar/DATASETS/2_digit_Mnist/test.csv")
train_images = os.getcwd() + "/2_digit_Mnist/train_images/" + train_df.iloc[:, 0].values
test_images = os.getcwd() + "/2_digits_Mnist/test_images/" + test_df.iloc[:, 0].values

train_labels = train_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 1:].values


def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)

    # In older versions you need to set shape in order to avoid error
    # on newer (2.3.0+) the following 3 lines can safely be removed
    image.set_shape((64, 64, 1))
    label[0].set_shape([])
    label[1].set_shape([])

    labels = {"first_num": label[0], "second_num": label[1]}
    return image, labels

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = (
    train_dataset.shuffle(buffer_size=len(train_labels))
    .map(read_image)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = (
    test_dataset.map(read_image)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

inputs = keras.Input(shape=(64, 64, 1)) # images are 64 by 64 with 1 color channel
x = layers.Conv2D(filters=32,kernel_size=3,padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(64, 3, kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, kernel_regularizer=regularizers.l2(WEIGHT_DECAY),activation='relu')(x)
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
output1 = layers.Dense(10, activation='softmax', name='first_num')(x) # buradaki isim ile csv file'daki isim ayni olmak zorunda
output2 = layers.Dense(10, activation='softmax', name='second_num')(x)

model = keras.Model(inputs=inputs, outputs=[output1, output2])

model.compile(loss=[
    keras.losses.SparseCategoricalCrossentropy(), # her bir output icin loss fuc tanimlanmali
    keras.losses.SparseCategoricalCrossentropy()], # eger bir tane tanimalasan bile ikisi icin de geceli olur ama saglikli olan ayri ayri yazmak
optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),metrics=['accuracy'])

model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=5, verbose=2)
model.evaluate(test_dataset, verbose=2)



