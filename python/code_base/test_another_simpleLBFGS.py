#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""An example of using tfp.optimizer.lbfgs_minimize to optimize a TensorFlow model.
This code shows a naive way to wrap a tf.keras.Model and optimize it with the L-BFGS
optimizer from TensorFlow Probability.
Python interpreter version: 3.6.9
TensorFlow version: 2.0.0
TensorFlow Probability version: 0.8.0
NumPy version: 1.17.2
Matplotlib version: 3.1.1
"""

# https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
import numpy
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import numpy as np
from tensorflow.keras.datasets import cifar10 # to load dataset

def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(model(train_x, training=True), train_y)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f

def plot_helper(inputs, outputs, title, fname):
    """Plot helper"""
    pyplot.figure()
    pyplot.tricontourf(inputs[:, 0], inputs[:, 1], outputs.flatten(), 100)
    pyplot.xlabel("x")
    pyplot.ylabel("y")
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.savefig(fname)

if __name__ == "__main__":

    # prepare training data
    # x_1d = numpy.linspace(-1., 1., 11)
    # x1, x2 = numpy.meshgrid(x_1d, x_1d)
    # x_train = numpy.stack((x1.flatten(), x2.flatten()), 1)
    # y_train = numpy.reshape(x_train[:, 0]**2+x_train[:, 1]**2, (x_1d.size**2, 1))
    
    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    
    # image_size = (64, 64)
    # img_height = 64
    # img_width = 64
    # batch_size = 200
    # # Load and explore dataset
    # allImages_path = "C:/Users/friedele/ImageData/TrainingData"
    # allImages_dir = pathlib.Path(allImages_path)
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     allImages_dir,
    #     #color_mode='grayscale',
    #     validation_split=0.2,
    #     labels='inferred',
    #     label_mode='categorical',
    #     subset="training",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)

    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #     allImages_dir,
    #     #color_mode='grayscale',
    #     validation_split=0.2,
    #     labels='inferred',
    #     label_mode='categorical',
    #     subset="validation",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)

    # class_names = train_ds.class_names
    # list_ds = tf.data.Dataset.list_files(allImages_path, shuffle=False)
    # for f in list_ds.take(5):
    #   print(f.numpy())

    # AUTOTUNE = tf.data.AUTOTUNE  # Optimize CPU usuage

    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # train_ds=list(train_ds)
    # val_ds=list(val_ds)

    # train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    # val_labels = np.concatenate([y for x, y in val_ds], axis=0)

    # updated_labels=[]
    # for i in range(int(train_labels.size/4)):
    #     for k in range(0,4):
    #         val=train_labels[i].item(k)
    #         if (val==1.0):
    #             updated_labels.append(k)
    # y_train = np.array(updated_labels, dtype=np.int32) 

    # updated_labels=[]
    # for i in range(int(val_labels.size/4)):
    #     for k in range(0,4):
    #         val=val_labels[i].item(k)
    #         if (val==1.0):
    #             updated_labels.append(k)
    # y_test = np.array(updated_labels, dtype=np.int32)   

    # y_train = train_labels
    # y_test = val_labels           
            
    # x_test = np.concatenate([val_ds[n][0] for n in range(0, len(val_ds))])
    # x_train = np.concatenate([train_ds[n][0] for n in range(0, len(train_ds))])

    # # Normalize
    # x_train = x_train / 255
    # x_test = x_test / 255

    # # prepare prediction model, loss function, and the function passed to L-BFGS solver
    # # pred_model = tf.keras.Sequential(
    # #     [tf.keras.Input(shape=[2,]),
    # #       tf.keras.layers.Dense(64, "tanh"),
    # #       tf.keras.layers.Dense(64, "tanh"),
    # #       tf.keras.layers.Dense(1, None)])
    # # Using my predicted model
    # img_height = 64
    # img_width = 64
    # num_classes = 4
    
    pred_model = Sequential([
        #layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
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
        layers.Dense(10)
    ])
    loss_fun = tf.keras.losses.MeanSquaredError()
    func = function_factory(pred_model, loss_fun, x_train, y_train)

    # convert initial model parameters to a 1D tf.Tensor
    init_params = tf.dynamic_stitch(func.idx, pred_model.trainable_variables)

    # train the model with L-BFGS solver
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func, initial_position=init_params, max_iterations=500)

    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    func.assign_new_model_parameters(results.position)

    # do some prediction
    pred_outs = pred_model.predict(x_train)
    err = numpy.abs(pred_outs-y_train)
    print("L2-error norm: {}".format(numpy.linalg.norm(err)/numpy.sqrt(11)))

    # plot figures
    plot_helper(x_train, y_train, "Exact solution", "ext_soln.png")
    plot_helper(x_train, pred_outs, "Predicted solution", "pred_soln.png")
    plot_helper(x_train, err, "Absolute error", "abs_err.png")
    pyplot.show()

    # print out history
    print("\n"+"="*80)
    print("History")
    print("="*80)
    print(*func.history, sep='\n')


