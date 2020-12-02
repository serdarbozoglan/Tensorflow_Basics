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

# 1. How to save and load model weights
# 2. How to save and load entire model (Serializing model)
   # Save the weights
   # Model architecture
   # Training configuration (model.compile())
   # Optimizer and states


model1 = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
    ]
)

inputs = keras.Input(784)
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor):
        x  = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)

model3 = MyModel()

model = model1
model.load_weights('saved_model/') # hangi modelin weightlerini save etmissek onu model olarak load edebiliriz yani 
# eger model2 yi save etmissek model1 icin o weightleri load edemeyiz
# eger sadece model.load_weights() yaparsan training configuration olan model.compile uncommented yapilmalidir
# model.compile(
#         optimizer = keras.optimizers.Adam(),
#         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy']
# )

model = keras.models.load_model('complete_saved_model/') # eger modeli yukluyorsak yukaridaki model tanimlamalirina ve compile fonksiyonuna da ihtiyac duyulmaz

model.fit(X_train, y_train,batch_size=64, epochs=20, verbose=1)
model.evaluate(X_test, y_test, batch_size=64, verbose=1)
model.save_weights('saved_model/') # just saves the weights
model.save('complete_saved_model/') # modeli buradan yukledigimizde model.compile() a gerek kalmaz olmadan da calisir
