import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras import models
import pathlib
import datetime as dt
import pickle
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
import levenberg_marquardt as lm

# Dataset Parameters
allImages_path = "C:/Users/friedele/Repos/DRFM/images/allTgts" 
allImages_dir = pathlib.Path(allImages_path)
rangeImages_path =  "C:/Users/friedele/Repos/DRFM/images/rangeTgts"
rangeImages_dir =  pathlib.Path(rangeImages_path)
dopImages_path =  "C:/Users/friedele/Repos/DRFM/images/dopTgts"
dopImages_dir =  pathlib.Path(dopImages_path)
randomImages_path =  "C:/Users/friedele/Repos/DRFM/images/randomTgts"
randomImages_dir =  pathlib.Path(randomImages_path)
combinedImages_path =  "C:/Users/friedele/Repos/DRFM/images/combinedTgts"
combinedImages_dir =  pathlib.Path(combinedImages_path)

image_size=(49,52)
img_height = 49
img_width = 52
batch_size=16
num_classes=4
epochs=32 # Why this value?
plotAdam = 1
plotLM = 0
runLM = 0
runAdam = 1 

# Load and explore dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  allImages_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  allImages_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
list_ds = tf.data.Dataset.list_files(allImages_path, shuffle=False)
for f in list_ds.take(5):
  print(f.numpy())

AUTOTUNE = tf.data.AUTOTUNE  # Optimize CPU usuage 

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

# Model 
# Model 
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  # How do we know how many times to Conv2D and maxPooling2D, here its 3 layers.
  layers.Conv2D(16, 4, padding='same', activation='relu',use_bias=True),  # Why did we pick these values?
  layers.MaxPooling2D(),
  layers.Conv2D(32, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # What is the best dropout rate?
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
# Training the Model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

def load_image(filename):
 # load the image
 img_height = 49
 img_width = 52
 img = load_img(filename, color_mode = "grayscale", target_size=(img_height, img_width))
 # convert to array
 img_tensor = img_to_array(img)
 img_tensor = np.expand_dims(img_tensor, axis=0)
 img_tensor /= 255.
 print(img_tensor.shape)
 plt.imshow(img_tensor[0])
 plt.show()
 return img_tensor

img_tensor = load_image('C:/Users/friedele/Repos/DRFM/images/dopTgts/n03-dopTgts/n03_15.png')

layer_outputs = [layer.output for layer in model.layers[:8]]
model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
