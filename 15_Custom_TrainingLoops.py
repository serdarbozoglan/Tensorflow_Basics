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

num_epochs = 5
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()

num_epochs = 5
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Training Loop
for epoch in range(num_epochs):
    print(f"\nStart of Training Epoch {epoch}")
    for batch_idx, (x_batch, y_batch) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y_batch, y_pred)

    train_acc = acc_metric.result()
    print(f"Accuracy over epoch {train_acc}")
    acc_metric.reset_states()

# Test Loop
for batch_idx, (x_batch, y_batch) in enumerate(ds_test):
    y_pred = model(x_batch, training=True)
    acc_metric.update_state(y_batch, y_pred)

train_acc = acc_metric.result()
print(f"Accuracy over Test Set: {train_acc}")
acc_metric.reset_states()

