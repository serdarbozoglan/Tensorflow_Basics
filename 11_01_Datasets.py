import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"], # somedatset has validation as well
    shuffle_files = True,
    as_supervised = True, #(image, label) otherwise it will return a dictionary
    with_info = True, # for ds_info
)

#fig = tfds.show_examples(ds_train, ds_info, rows=4, cols=4)
#print(ds_info)

def normalize_image(image, label):
    return tf.cast(image, tf.float32)/255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

# Asagidaki islemleri kendi custom  datasetimiz icin de yapacagiz
ds_train = ds_train.map(normalize_image, num_parallel_calls=AUTOTUNE) # this appyling func should be in sequential but can be done in parallel
# so we can define how many parallel we can do it. AUTOTUNE will automatically be doing it for us 
ds_train = ds_train.cache() # after first time loaded the data, it will be faster for the next time
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples) # shuffles the data
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# We are NOT shuffling ds_test
ds_test = ds_test.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)

model = keras.Sequential([
        keras.Input((28,28,1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(10)
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])

model.fit(ds_train, epochs=5, verbose=1)
model.evaluate(ds_test, verbose=1)



