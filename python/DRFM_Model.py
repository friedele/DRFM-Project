import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import datetime as dt
import pickle
import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import levenberg_marquardt as lm

# Dataset Parameters
allImages_path = "C:/Users/friedele/Repos/DRFM/images/allTgts"
allImages_dir = pathlib.Path(allImages_path)
rangeImages_path = "C:/Users/friedele/Repos/DRFM/images/rangeTgts"
rangeImages_dir = pathlib.Path(rangeImages_path)
dopImages_path = "C:/Users/friedele/Repos/DRFM/images/dopTgts"
dopImages_dir = pathlib.Path(dopImages_path)
randomImages_path = "C:/Users/friedele/Repos/DRFM/images/randomTgts"
randomImages_dir = pathlib.Path(randomImages_path)
combinedImages_path = "C:/Users/friedele/Repos/DRFM/images/combinedTgts"
combinedImages_dir = pathlib.Path(combinedImages_path)

image_size = (49, 52)
img_height = 49
img_width = 52
batch_size = 16
num_classes = 4
epochs = 32  # Why this value?
plotAdam = 0
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
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    # How do we know how many times to Conv2D and maxPooling2D, here its 3 layers.
    layers.Conv2D(16, 4, padding='same', activation='relu',
                  use_bias=True),  # Why did we pick these values?
    layers.MaxPooling2D(),
    layers.Conv2D(32, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),  # What is the best dropout rate?
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes)
])

if runAdam:
  # Compile Model Optimizer
    model.compile(optimizer='adam',
                  # loss=tf.keras.losses.CategoricalHinge(from_logits=True),
                  #loss=tf.keras.losses.MeanSquaredError(),
                  #loss=tf.keras.losses.CategoricalCrossentropy,
                  #loss=tf.keras.losses.BinaryCrossentropy(),
                  #loss=tf.keras.losses.Huber(),
                  #loss=tf.keras.losses.SquaredHinge(),
                  #loss=tf.keras.losses.CosineSimilarity(),
                  #loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    # Instantiate an optimizer.
    Optimizer = tf.keras.optimizers.Adam()
    Optimizer.variables()

    # Train the Model
    history = model.fit(train_ds, validation_data=val_ds, epochs=30)

    # Prediction based on truth data
    adam_prediction = model.predict(train_ds)

    # Generate arg maxes for predictions
    classes = np.argmax(adam_prediction, axis=1)

if runLM:
    model_LM = lm.ModelWrapper(
        tf.keras.models.clone_model(model))

    model_LM.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        #  optimizer = tf.keras.optimizers.SGD(lr=1.0, momentum=0.1, nesterov=True), #Damping factor issue
        #  optimizer = tf.keras.optimizers.RMSprop(learning_rate=1),
        solver='qr',  # qr, cholesky, solve
        loss=lm.MeanSquaredError())
    #loss=lm.BinaryCrossentropy())
    #loss=lm.CategoricalMeanSquaredError())
    #loss=lm.ReducedOutputsMeanSquaredError())
    #loss=lm.SparseCategoricalCrossentropy())
    #loss=lm.CategoricalCrossentropy()) Errors out

    # Training the Model
    history = model_LM.fit(train_ds, validation_data=val_ds, epochs=10)

# if runSGD:
#     tf.keras.optimizers.SGD(
#     learning_rate=0.01,
#     momentum=0.0,
#     nesterov=False,
#     name="SGD",
#     **kwargs
#     )

### SAVE MODEL ###
# saving the model
model.save("C:/Users/friedele/Repos/DRFM/ouputs/trainedmodel_20Epoch.h5")
with open('trainHistoryOld', 'wb') as handle:  # saving the history of the model to CoLab /content
    pickle.dump(history.history, handle)

with open('C:/Users/friedele/Repos/DRFM/ouputs/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open('C:/Users/friedele/Repos/DRFM/ouputs/trainHistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)

### Plot Model Metric Results ###
if plotAdam:
    path = "C:/Users/friedele/Repos/DRFM/ouputs/"
    date_string = dt.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    acc_file = date_string + "_accuracy"
    val_acc_file = date_string + "_validation_accuracy"

    acc = (history['accuracy'])
    valAcc = (history['val_accuracy'])
    scipy.io.savemat(os.path.join(path, acc_file), mdict={'acc': acc})
    scipy.io.savemat(os.path.join(path, val_acc_file),
                     mdict={'valAcc': valAcc})
    plt.plot(acc)  # Trained Data
    plt.plot(valAcc)  # Model Validation
    plt.title('Adam: Model Relative Error')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid(True)
    plt.show()

    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.grid(True)
    # plt.show()

    # pd.DataFrame(history).plot(figsize=(8,5))
    # plt.show()
