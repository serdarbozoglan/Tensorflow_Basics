import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub


# # To Avoid GPU errors
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ================================================ #
#                  Our Pretrained-Model            #
# ================================================ #

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# model = keras.models.load_model("pretrained")

# # Freeze all model layer weights
# model.trainable = False

# # Can also set trainable for specific layers
# for layer in model.layers:
#     # assert should be true because of one-liner above
#     assert layer.trainable == False
#     layer.trainable = False

# print(model.summary())  # for finding base input and output
# base_inputs = model.layers[0].input # pre-trained modelin input'u kendi modelimiz icinde input layer olut
# base_output = model.layers[-2].output # son layer'dan onceki layeri base output olarak aliriz, cunku son layer isimize yaramayabilir
# # ornegin imagenet uzerinde train edilmis bir modelin num_classess i 1000 olabilir ama bize 5 class lazimsa son layeri ignore ederiz
# output = layers.Dense(10)(base_output) # kendi custom son layerimizi ekleriz
# new_model = keras.Model(base_inputs, output) # custom model

# # This model is actually identical to model we
# # loaded (this is just for demonstration and
# # and not something you would do in practice).
# print(new_model.summary())

# # As usual we do compile and fit, this time on new_model
# new_model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# new_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)

# ================================================ #
#                Pretrained Keras Model            #
# ================================================ #

# https://www.tensorflow.org/api_docs/python/tf/keras/applications  --> keras modelleri burada gorulebilir
# X = tf.random.normal(shape=(5, 299, 299, 3)) # we have 5,  299 by 299 RGB fictious images
# y = tf.constant([0, 1, 2, 3, 4]) # we have 5 classes, each image has a unique class

# model = keras.applications.InceptionV3(include_top=True) # include_top=False yaparsan son layer'i exclude eder
# base_inputs = model.layers[0].input
# base_outputs = model.layers[-2].output # son layer dahil edildiginde (include_top=True), en sondan bir oncelki layer bizim output layer imiz olur
# final_outputs = layers.Dense(5)(base_outputs)

# new_model = keras.Model(inputs=base_inputs, outputs=base_outputs)

# new_model.compile(
#         optimizer = keras.optimizers.Adam(),
#         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy']
# )

# new_model.fit(X, y, epochs=5, batch_size=64, verbose=1)

# ================================================ #
#               Tensorflow Hub        #
# ================================================ #

# https://tfhub.dev
X = tf.random.normal(shape=(5, 299, 299, 3)) # we have 5,  299 by 299 RGB fictious images
y = tf.constant([0, 1, 2, 3, 4]) # we have 5 classes, each image has a unique class

# feature vector model'in Fully Connected DEnse Layer'lari exclude edilmis halidir
url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"

base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
base_model.trainable = False

new_model = keras.Sequential([
            base_model, 
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(5)
])

new_model.compile(
        optimizer = keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
)

new_model.fit(X, y, epochs=5, batch_size=64)





