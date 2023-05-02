import tensorflow as tf
import random
# libraries for data manipulation
import numpy as np
import pandas as pd	
import pathlib

# libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Replace with my data
image_size = (64, 64)
img_height = 64
img_width = 64
batch_size = 1

# Load and explore dataset
allImages_path = "C:/Users/friedele/ImageData/TrainingData"
allImages_dir = pathlib.Path(allImages_path)
train_ds = tf.keras.utils.image_dataset_from_directory(
    allImages_dir,
    color_mode='grayscale',
    validation_split=0.2,
    labels='inferred',
    label_mode='categorical',
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    allImages_dir,
    color_mode='grayscale',
    validation_split=0.2,
    labels='inferred',
    label_mode='categorical',
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

train_ds=list(train_ds)
val_ds=list(val_ds)

train_labels = np.concatenate([y for x, y in train_ds], axis=0)
val_labels = np.concatenate([y for x, y in val_ds], axis=0)

updated_labels=[]
for i in range(int(train_labels.size/4)):
    for k in range(0,4):
        val=train_labels[i].item(k)
        if (val==1.0):
            updated_labels.append(k)
y_train = np.array(updated_labels, dtype=np.uint8) # hot encoded labels

updated_labels=[]
for i in range(int(val_labels.size/4)):
    for k in range(0,4):
        val=val_labels[i].item(k)
        if (val==1.0):
            updated_labels.append(k)
y_test = np.array(updated_labels, dtype=np.uint8)              
        
val_ds = np.concatenate([val_ds[n][0] for n in range(0, len(val_ds))])
train_ds = np.concatenate([train_ds[n][0] for n in range(0, len(train_ds))])

x_train = train_ds
x_test = val_ds

# Normalizing the data by divinding it with 255
# The reason is because the pixel values of the image can vary between 0-255
x_train, x_test = x_train / 255.0, x_test / 255.0

mnist_train = pd.DataFrame(x_train.reshape(x_train.shape[0], 
                                           x_train.shape[1] * x_train.shape[2]))
# repeat the same for test data
mnist_test = pd.DataFrame(x_test.reshape(x_test.shape[0], 
                                           x_test.shape[1] * x_test.shape[2]))

# Create column names for dataframe to make it more readable
colnames = ['Pixel'+str(i) for i in range(1,4097)]

mnist_train.columns = colnames
mnist_test.columns = colnames

# concatenate the labels into the test and train dataframe
mnist_train['label'] = y_train
mnist_test['label'] = y_test

mnist_train.head()
# check if there is any NaN or missing datapoints
mnist_train.isnull().values.sum()
sns.countplot(x="label", data=mnist_test, palette="Set2").set(
    title="Label distribution in Test Data")
sns.countplot(x="label", data=mnist_train, palette="Set3").set(
    title="Label distribution in Training Data")

# visualise some of the images from train dataset
# display 4 images per row
fig, axes = plt.subplots(3,4, figsize=(10,5))
# set a tight layout for better spacing between plots
fig.tight_layout()
axes = axes.flatten()
# generate 12 random row numbers to select from mnist_train dataframe
idx = np.random.randint(0,mnist_train.shape[0],size=12)
for i in range(12):
    # get the row data
    pixel_data = mnist_train.iloc[idx[i]]
    # use imshow to build the image using pixel info
    # reshape is necessary because imshow expects a 2d array structure as input
    axes[i].imshow(np.array(pixel_data[:4096]).reshape(64,64), cmap='gray')
    axes[i].axis('off') 
    # add the label of the image as title
    axes[i].set_title(str(int(pixel_data[-1])), color= 'black', fontsize=20)
    
import xgboost as xb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# splitting the training data into train and validate datasets
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train, 
                                                  test_size=0.1)

# All the numpy array structures need to be transformed to DMatrix structure
# which is optimized specifically for XGBoost models
# reshaping the x_train, x_val and x_test to a 2d matrix
xgb_train_matrix = xb.DMatrix(x_train.reshape(x_train.shape[0], 
                                    x_train.shape[1] * x_train.shape[2]), 
                              label=y_train)

xgb_val_matrix = xb.DMatrix(x_val.reshape(x_val.shape[0], 
                                    x_val.shape[1] * x_val.shape[2]), 
                              label=y_val)

xgb_test_matrix = xb.DMatrix(x_test.reshape(x_test.shape[0], 
                                    x_test.shape[1] * x_test.shape[2]), 
                              label=y_test)

# set the hyperparameters for the model
params = {
    'max_depth': 10,                # the maximum depth of each tree
    'eta': 0.7,                     # the training step for each iteration
    'objective': 'multi:softmax',   # multiclass classification using the softmax objective
    'num_class': 4,                # labels range from 0-9 hence the num_classes is 10
    'eval_metric': ['merror' ,      # evaluation metric as mean squared error
                    'mlogloss']     # and log loss
}  

results = {}
xgb_model = xb.train(params, xgb_train_matrix, evals=[(xgb_train_matrix, 'train'),
                                                      (xgb_val_matrix, 'val')],
                     num_boost_round=10,
                     evals_result = results,
                     verbose_eval=True)

y_pred = xgb_model.predict(xgb_test_matrix)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))   

epochs = range(len(results['train']['merror']))
plt.plot(epochs,results['train']['merror'], color='green', label='Train' )
plt.plot(epochs,results['val']['merror'], color='red', label='Validation' )
plt.title('XGBoost Classification error')
plt.xlabel('epoch')
plt.ylabel('Classification error')
plt.legend()
plt.show()

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from keras import optimizers
from keras import models

# add a channel dimension to the images
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_val= np.expand_dims(x_val, axis=-1)

# one hot encoding all the labels
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
y_val = to_categorical(y_val,10)

cnn_model = models.Sequential()

cnn_model.add(Conv2D(filters=20, kernel_size=(5, 5), 
                   activation='relu', padding="same", 
                   input_shape=(64,64,1)))
cnn_model.add(BatchNormalization(axis=-1))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(filters=20, kernel_size=(4, 4), 
                   activation='relu', padding="same"))
cnn_model.add(BatchNormalization(axis=-1))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(filters=20, kernel_size=(4, 4), 
                   activation='relu', padding="same"))
cnn_model.add(BatchNormalization(axis=-1))
cnn_model.add(Dropout(0.2))

cnn_model.add(Flatten())
cnn_model.add(Dense(200, activation='relu'))

cnn_model.add(Dense(10, activation='softmax'))

cnn_model_opt = optimizers.Adam(decay=1e-4)

print(cnn_model.summary())

cnn_model.compile(optimizer = cnn_model_opt , loss = "categorical_crossentropy", 
                  metrics=["accuracy"]) 

cnn_model_fit = cnn_model.fit(x_train, 
                        y_train,
                        validation_data = (x_val, y_val),
                        batch_size=128,
                        epochs=10)

cnn_test_accuracy = cnn_model.evaluate(x_test, y_test)
print("Accuracy on test data is : ", cnn_test_accuracy[1] * 100)