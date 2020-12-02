import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text

(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"], # somedatset has validation as well
    shuffle_files = True,
    as_supervised = True, #(image, label) otherwise it will return a dictionary
    with_info = True, # for ds_info
)

#print(ds_info)
# for text, label in ds_train:
#     print(text)
#     import sys
#     sys.exit()

# TOKENIZE
# CONVERT TOKENS INTO INT

tokenizer = tfds.features.text.tokenizer() # tensorflow_text.WhitespaceTokenizer() # tfds.features.text.tokenizer()

def build_vocabulary():
    vocabulary = set()
    for text, _ in ds_train:
        vocabulary.update(tokenizer.tokenize(text.numpy().lower()))  
    return vocabulary # now we add all words into vocabulary but maybe we should add only the ones which occcur more than a certain limit

vocabulary = build_vocabulary()

encoder = tfds.features.text.TokenTextEncoder(
        vocabulary, oov_token="<UNK>", lowercase=True, tokenizer=tokenizer 
)

def my_encoding(text_tensor, label):
    return encoder.encode(text_tensor.numpy()), label # with encoder we tokenize and then encode it

def encode_map(text, label):
    encoded_text, label = tf.py_function(
        my_encoding, inp=[text, label], Tout=(tf.int64, tf.int64)
    )

    encoded_text.set_shape([None]) # Text can be with arbitrary length
    label.set_shape([]) # will be a single integer so it is a scalar

    return encoded_text, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_map, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ())) # None yazdigimiz kisim pad edilecek kisim. Ynei version tf'te bu kisma gerek yok

ds_test = ds_test.map(encode_map)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

model = keras.Sequential([
    layers.Masking(mask_value=0), # "0" is padding value and will be ignored in computations
    layers.Embedding(input_dim=len(vocabulary)+2, output_dim=32), # plus 1 for padding value(0) and we have <UNK> (oov_words) as well, 
    # output dim is 32 dimentional embedding whihc is very small actually
    # each word is converted into 32 dimentional vector
    # BATCH_SIZE x 1000 --> BATCH_SIZE x 1000 X 32
    layers.GlobalAvgPool1D(),  # burada batch size 1000 (yani 1000 sequential word'un 32 dimention'lik vectorunun average'ini aliriz)
    # GlobalAvgPool sonucunda cikan shape BATCH_SIZE X 32 
    layers.Dense(64, activation='relu'),
    layers.Dense(1) # it is a binary classification positive or negative review
    ]
)

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(3e-4, clipnorm=1), # we clip the gradients so that we do not have expoloding gradients
              metrics=['accuracy'])

model.fit(ds_train, epochs=10, verbose=2)
model.evaluate(ds_test, verbose=1)