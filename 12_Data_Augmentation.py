import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
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

# Data augmentation icin 1.method asagidakidir
# Bu method'da model train edilirken bir taraftan da augmentation yapilir on the fly (CPU'da olur augmentation while training happens in  GPU (if available))
# Every image is augmented sequentially and "on the fly", so we increase our dataset although we do it implicitly. 
# Let's say we use a random rotation of -30 to 30 degrees of the original image, then it's difficult to say how much we've increased 
# our dataset. We do not store anything to disk but every iteration the image will be randomly rotated between [-30, 30] degrees, 
# and in theory this would be infinitely many different images.
def augment(image, label):
    new_height = new_width = 32 # datasettte zaten 32 by 32 olarak verilmis durumda ama olmasa boyle yapabilirdik
    image = tf.image.resize(image, (new_height, new_width))

    if tf.random.uniform((), minval=0, maxval=1) < 0.1: # image'larin bir kismini (10%) gray scale'e cevirecegiz ILAVE olarak
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3]) # output channel became 1 channel now and model would expect 3 channels we will solve this now by tf.tile
        # so we're just coypying the 3rd dimensin 3 times here and we do not toch first and second dimesnions (height and width)
    
    image = tf.image.random_brightness(image, max_delta=0.1) # adding radom brightness
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2) 

    # when flipping the image you should be very carefull
    # you do not want to flip 6 to 9 or vice verse. Image can be altered completely to new class so be careful
    # data augmentation should not destroy the current labels
    # more data augmentation is NOT always better
    image = tf.image.random_flip_left_right(image) # 50%
    # image = tf.image.random_flip_up_down # 50%

    return image, label


# Asagidaki islemleri kendi custom  datasetimiz icin de yapacagiz
ds_train = ds_train.map(normalize_image, num_parallel_calls=AUTOTUNE) # this appyling func should be in sequential but can be done in parallel
# so we can define how many parallel we can do it. AUTOTUNE will automatically be doing it for us 
ds_train = ds_train.cache() # after first time loaded the data, it will be faster for the next time
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples) # shuffles the data
#ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# We are NOT shuffling and cahcing in ds_test
ds_test = ds_test.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)

# TF >= 2.3.0 ise asagidaki sekilde Data Augmentation yapilabilir
# Bu method'da data augmentation modelin bir parcasidir
# Yukaridaki gibi paralel bir islem olmadigi icin performans kaybi olabilir ama daha simpledir
data_augmentation  = keras.Sequential([
            layers.experimental.preprocessing.Resizing(height=32, width=32),
            layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
            layers.experimental.preprocessing.RandomContrast(factor=0.1),
]) 

model = keras.Sequential([
        keras.Input((32, 32 ,3)),
        data_augmentation, # 2. data augmentation method devrede iken kullanilir
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(lr=3e-4),metrics=['accuracy'])

model.fit(ds_train, epochs=5, verbose=1)
model.evaluate(ds_test, verbose=1)



