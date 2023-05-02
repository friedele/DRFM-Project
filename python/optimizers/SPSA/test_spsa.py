import time
import tensorflow.compat.v1 as tf
import spsa
import levenberg_marquardt as lm

logdir = 'logs/run_{}'.format(time.time())

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.cast(x_train / 255.0, dtype=tf.float32)
x_test = tf.cast(x_test / 255.0, dtype=tf.float32)

x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

y_train = tf.cast(y_train, dtype=tf.float32)
y_test = tf.cast(y_test, dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000)
train_dataset = train_dataset.batch(6000).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# common training parameters
training_epochs = 5
batch_size = 100

# Gradient Descent parameters
lr = 0.01

# SPSA parameters
a = 0.01
c = 0.01
alpha = 1.0
gamma = 0.4


x = tf.placeholder(tf.float32, [None, 784], name='Input') 
y = tf.placeholder(tf.float32, [None, 10], name='Labels') 

# model
with tf.name_scope('Model'):
    W = tf.Variable(tf.random_uniform([784, 10]), name='Weights')
    b = tf.Variable(tf.random_uniform([10]), name='Bias')
    p = tf.nn.softmax(tf.matmul(x, W) + b)

# objective
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(p), reduction_indices=1))
# def _minimize_spsa(self, lr, num_iterations, alpha, gamma, loss):
with tf.name_scope('Optimizer'):
  #  optimizer = spsa.SPSA()._minimize_spsa(lr,10,alpha,gamma,cost)
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),

  

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Variables for Tensorboard
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", acc)
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    # Training Loop
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op, cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                      feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))


