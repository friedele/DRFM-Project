from numpy import argmax
import keras
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from tensorflow.keras import layers
import pathlib

# load and prepare the image
# def load_image(filename):
#  # load the image
#  img = load_img(filename, grayscale=True, target_size=(28, 28))
#  # convert to array
#  img = img_to_array(img)
#  # reshape into a single sample with 1 channel
#  img = img.reshape(1, 28, 28, 1)
#  # prepare pixel data
#  img = img.astype('float32')
#  img = img / 255.0
#  return img

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
def run_example():
 # load the image
 img = load_image('C:/Users/friedele/Repos/DRFM/images/dopTgts/n03-dopTgts/n03_2.png')
 # load model
 model = load_model('C:/Users/friedele/Repos/DRFM/ouputs/trainedmodel_20Epoch.h5')
 # predict the class
 predict_value = model.predict(val_ds)
 digit = argmax(predict_value)
 print(digit)
 
# entry point, run the example
run_example()