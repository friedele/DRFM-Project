import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import levenberg_marquardt as lm
import spsa

input_size = 20000
batch_size = 1000
#conda install spyder=5.5.3
x_train = np.linspace(-1, 1, input_size, dtype=np.float64)
y_train = np.sinc(10 * x_train)

x_train = tf.expand_dims(tf.cast(x_train, tf.float32), axis=-1)
y_train = tf.expand_dims(tf.cast(y_train, tf.float32), axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(input_size)
train_dataset = train_dataset.batch(batch_size).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(40, activation='tanh', input_shape=(1,)),  # Better results
    tf.keras.layers.Dense(1, activation='linear')])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError())

model_LM = lm.ModelWrapper(
    tf.keras.models.clone_model(model))

model_LM.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
  #  optimizer = tf.keras.optimizers.SGD(lr=1.0, momentum=0.9, nesterov=False),
    loss=lm.MeanSquaredError())

model_spsa = spsa.ModelWrapper(
    tf.keras.models.clone_model(model))

model_spsa.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=lm.MeanSquaredError())

# print("Train using SPSA")
# t1_start = time.perf_counter()
# model_spsa.fit(train_dataset, epochs=10)
# t1_stop = time.perf_counter()
# print("Elapsed time: ", t1_stop - t1_start)

print("\n_________________________________________________________________")
print("Train using Levenberg-Marquardt")
t2_start = time.perf_counter()
model_LM.fit(train_dataset, epochs=10)
t2_stop = time.perf_counter()
print("Elapsed time: ", t2_stop - t2_start)

lm_pred = model_LM.predict(train_dataset)
# Generate arg maxes for predictions
classes = np.argmax(lm_pred, axis=1)
print(classes)

print("\n_________________________________________________________________")
print("Plot results")
plt.plot(x_train, y_train, 'b-', label="reference")
plt.plot(x_train, model.predict(x_train), 'g--', label="adam")
plt.plot(x_train, model_LM.predict(x_train), 'r--', label="lm")
# plt.plot(x_train, model_SPSA.predict(x_train), 'k--', label="spsa")
plt.title('Sinc Func (Adam vs Levenberg-Marquardt)')
plt.legend()
plt.grid(True)
plt.show()