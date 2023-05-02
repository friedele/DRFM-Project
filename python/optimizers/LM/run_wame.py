from tensorflow import keras
from wame import WAME
from spsa import SPSA
from adam import ADAM
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


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
plotAdam = 0
plotLM = 0
runLM = 1
runAdam = 0 

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

model = Sequential([
  layers.Rescaling(1./255),
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

# This does not seem to optimize at all just repeats!
model.compile(optimizer=ADAM(learning_rate=0.001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=20) 