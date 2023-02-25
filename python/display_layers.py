from numpy import argmax
import keras
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from tensorflow.keras import layers
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# load and prepare the image
def load_image(filename):
 # load the image
 img_height = 49
 img_width = 52
 img = load_img(filename, color_mode = "grayscale", target_size=(img_height, img_width))
 # convert to array
 img = img_to_array(img)
 # reshape into a single sample with 1 channel
 #img = img.reshape(1, 49, 52, 1)
 img = layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
 return img

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
img_height = 49
img_width = 52
batch_size=16

val_ds = tf.keras.utils.image_dataset_from_directory(
  rangeImages_dir,
  validation_split=0.010, # Uses 1 file
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
 
# load an image and predict the class
def load_image(filename):
 # load the image
 img_height = 64
 img_width = 64
 img = load_img(filename, color_mode = "grayscale", target_size=(img_height, img_width))
 # convert to array
 img_tensor = img_to_array(img)
 img_tensor = np.expand_dims(img_tensor, axis=0)
 img_tensor /= 255.
 print(img_tensor.shape)
 plt.imshow(img_tensor[0])
 plt.show()
 return img_tensor
 
 
def run_example():
 # load the image
 from keras import models
 img_tensor = load_image('C:/Users/friedele/Repos/DRFM/images/tmpImage.png')
 # load model
 model = load_model('C:/Users/friedele/Repos/DRFM/ouputs/trainedmodel_50Epoch.h5')
 # predict the class
 # print(img_tensor.shape)
 # plt.imshow(img_tensor[0])
 # plt.show() 
 model.summary()
 layer_outputs = [layer.output for layer in model.layers[:8]]
 model = models.Model(inputs=model.input, outputs=layer_outputs)
 activations = model.predict(img_tensor)
 first_layer_activation = activations[0]
 print(first_layer_activation.shape)
 
 
### Main Program 
run_example()

