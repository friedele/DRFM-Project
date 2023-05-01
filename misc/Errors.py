"""
ValueError: Could not interpret optimizer identifier: <spsa.SPSA object at 0x0000016115ACDA00>

The reason is you are using tensorflow.python.keras API for model and layers and 
keras.optimizers for SGD. They are two different Keras versions of TensorFlow and pure Keras. 
They could not work together. You have to change everything to one version. Then it should work.





"""