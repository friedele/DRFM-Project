import copy
import sys
import tensorflow as tf

from neuralnetwork.activationfunctions.linear import Linear

sys.path.insert(0, "../../")
sys.path.insert(0, "./")
from os import makedirs
from neuralnetwork.model.neuralnetwork import NeuralNetwork
from neuralnetwork.optimizer import LBFGS
from neuralnetwork.datasets.dataset_utils import load_monk_dataset, load_cup
from neuralnetwork.utils.utils import  print_results_to_table_array_of_model
import numpy as np

# use float64 by default
tf.keras.backend.set_floatx("float64")
input_size = 20000
batch_size = 1000
#conda install spyder=5.5.3
x_train = np.linspace(-1, 1, input_size, dtype=np.float64)
y_train = np.sinc(10 * x_train)

# x_train = tf.expand_dims(tf.cast(x_train, tf.float32), axis=-1)
# y_train = tf.expand_dims(tf.cast(y_train, tf.float32), axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(input_size)
train_dataset = train_dataset.batch(batch_size).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

def main():
    np.random.seed(seed=123)
    for monk in [1]:

        if monk != 4:
            print("Loading datasets...")
            X_train, Y_train, X_test, Y_test = load_monk_dataset(monk)
            ds = "results/M_optimization/Monk{}".format(monk)
            threshold = 0.01
        elif monk == 4:
            print("Loading datasets...")
            X_train, Y_train = load_cup()
            ds = "results/M_optimization/CUP"
            threshold = 1.0
        makedirs(ds, exist_ok=True)



        models = []
        ms =  [3, 5, 7, 10, 15, 20, 25, 30]
        for m in ms:
            print("Building LBFGS model ... ")

            model = NeuralNetwork(LBFGS(m=m, c1=1e-4, c2=0.9, threshold_gradient_norm=0.00001, threshold_loss=threshold), regulariz_lambda=0.00001, loss_type="regularized_mse")
            if monk != 4:
                model.add_layer(4, features = 17)
                model.add_layer(1)
            elif monk == 4:
                model.add_layer(100, features=X_train.shape[1])
                model.add_layer(100)
                model.add_layer(2, activation=Linear())
            try:
                model.learn(X_train, Y_train, iterations=500)
                models.append(copy.deepcopy(model))
            except Exception as e:
                print(e)
                continue

        # Print plots
        print_results_to_table_array_of_model(models,ds, ms)
if __name__ == '__main__':
    main()
