import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import datetime as dt
import pickle
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
import levenberg_marquardt as lm
from diffgrad import DiffGrad 
from aadam import AAdam

# Dataset Parameters
allImages_path = "C:/Users/friedele/Repos/allDogs" 
allImages_dir = pathlib.Path(allImages_path)
beagleImages_path =  "C:/Users/friedele/Repos/beagle"
beagleImages_dir =  pathlib.Path(beagleImages_path)
goldenImages_path =  "C:/Users/friedele/Repos/golden"
goldenImages_dir =  pathlib.Path(goldenImages_path)
pomImages_path =  "C:/Users/friedele/Repos/pom"
pomImages_dir =  pathlib.Path(pomImages_path)

batch_size=16
plotAdam = 0
plotLM = 1
runLM = 0
runAdam = 1 

# Load and explore dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  allImages_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
allImages_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  batch_size=batch_size)

class_names = train_ds.class_names
list_ds = tf.data.Dataset.list_files(allImages_path, shuffle=False)
for f in list_ds.take(5):
  print(f.numpy())

AUTOTUNE = tf.data.AUTOTUNE  # Optimize CPU usuage 

train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Model 
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

# l =[DiffGrad(beta_1=1)]
# for opt in l:

model.compile(optimizer=AAdam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


if runAdam:
# # Compile Model Optimizer
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Instantiate an optimizer.
    Optimizer = tf.keras.optimizers.Adam()
    Optimizer.variables()
    
    # Train the Model
    history = model.fit(train_ds, validation_data=val_ds, epochs=1)
    
    # Prediction based on truth data
    adam_prediction = model.predict(train_ds)
    
    # Generate arg maxes for predictions
    classes = np.argmax(adam_prediction, axis = 1)


## Blows up as well!
if runLM:

    model_LM = lm.ModelWrapper(tf.keras.models.clone_model(model))

    model_LM.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=lm.SparseCategoricalCrossentropy(from_logits=True),
        solve_method='solve',
        metrics=['accuracy'])
    
    history = model_LM.fit(
      train_ds,
      validation_data=val_ds,
      epochs=20) # Why did we pick this value?)
    
    # Training the Model
    history = model_LM.fit(train_ds, validation_data=val_ds, epochs=20)

### SAVE MODEL ###
model.save("C:/Users/friedele/Repos/DRFM/ouputs/trainedmodelDogs_20Epoch.h5") # saving the model
with open('trainHistoryOld', 'wb') as handle: # saving the history of the model to CoLab /content
    pickle.dump(history.history, handle)

with open('C:/Users/friedele/Repos/DRFM/ouputs/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open('C:/Users/friedele/Repos/DRFM/ouputs/trainHistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)   
    
### Plot Model Metric Results ###
if plotAdam:
    path = "C:/Users/friedele/Repos/DRFM/ouputs/" 
    date_string  = dt.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    acc_file = date_string + "_accuracy"
    val_acc_file= date_string + "_validation_accuracy"
    
    acc=(history['accuracy']) 
    valAcc=(history['val_accuracy'])
    scipy.io.savemat(os.path.join (path, acc_file), mdict={'acc': acc})
    scipy.io.savemat(os.path.join (path, val_acc_file), mdict={'valAcc': valAcc})
    plt.plot(acc)  # Trained Data
    plt.plot(valAcc) # Model Validation
    plt.title('Adam: Model Relative Error')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid(True)
    plt.show() 

