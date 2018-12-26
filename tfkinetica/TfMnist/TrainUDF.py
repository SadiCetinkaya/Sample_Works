from __future__ import print_function
import KineticaIO
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from kinetica_proc import ProcData


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_NAME = "Mnist_2-256-layer"  # for tensorflow model name, always ends up with .ckpt
OUTPUT_NODE_NAMES = "input,output"
# TF Parameters
LEARNING_RATE = 0.1
MODEL_PATH = "/tmp/TFmodel/"  # Don't change this line
# Network Parameters
N_HIDDEN_1 = 256  # 1st layer number of neurons
N_HIDDEN_2 = 256  # 2nd layer number of neurons
N_INPUT = 784  # MNIST data input (img shape: 28*28)
N_CLASSES = 10  # MNIST total classes (0-9 digits)
# tf Graph input
X = tf.placeholder("float", [None, N_INPUT], name="input")  # place holder, None means it takes any batch size.
Y = tf.placeholder("float", [None, N_CLASSES])
# iterations
N_EPOCHS = 5
DISPLAY_STEP = 1
BATCH_SIZE = 100
NUM_EXAMPLES = 55000
ERROR_RATE_PRINTOUT_INTERVAL = 50  # by number of batches -> (NUM_EXAMPLES / BATCH_SIZE)


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name="output")
    return out_layer


# Store layers weight & bias
WEIGHTS = {
    'h1': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_1])),
    'h2': tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2])),
    'out': tf.Variable(tf.random_normal([N_HIDDEN_2, N_CLASSES]))
}
BIASES = {
    'b1': tf.Variable(tf.random_normal([N_HIDDEN_1])),
    'b2': tf.Variable(tf.random_normal([N_HIDDEN_2])),
    'out': tf.Variable(tf.random_normal([N_CLASSES]))
}
# Construct model
PREDICTOR = multilayer_perceptron(X, WEIGHTS, BIASES)
# Define loss and optimizer
COST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=PREDICTOR, labels=Y))
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(COST)
# Initialize the variables (i.e. assign their default value)
INIT = tf.global_variables_initializer()


def error_rate(predictions, labels):
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]
    error = 100.0 - (100 * float(correct) / float(total))
    return error


def main():
    proc_data = ProcData()
    kio = KineticaIO.KineticaIO()
    with tf.Session() as sess:
        sess.run(INIT)
        num_batches = int(NUM_EXAMPLES / BATCH_SIZE)
        print("{} batches with {} training examples per batch, "
              "makes total of {}.".format(num_batches, BATCH_SIZE, NUM_EXAMPLES))
        for epoch in range(N_EPOCHS):
            avg_cost = 0.0
            # Loop over all batches
            for batch_id in range(num_batches):
                td = pd.DataFrame(kio.get_data(offset=batch_id * BATCH_SIZE, number_data=BATCH_SIZE))
                data = np.array([np.fromstring(di, dtype=np.float32) for di in td['data']])
                label = np.array([np.fromstring(li) for li in td['onehot_label']])
                batch_x, batch_y = data, label
                _, c, predict = sess.run([OPTIMIZER, COST, PREDICTOR], feed_dict={X: batch_x, Y: batch_y})
                # Compute average loss
                if batch_id % ERROR_RATE_PRINTOUT_INTERVAL == 0:
                    print("the error rate at batch id " + str(batch_id) + " is " + str(error_rate(predict, label)))
                avg_cost += c / num_batches
            # Display cost per epoch step
            if epoch % DISPLAY_STEP == 0:
                print("Epoch: {}  Cost: {}".format(epoch + 1, avg_cost))
        print("Optimization Finished!")
        kio.model_to_kinetica(sess=sess, graph=tf.get_default_graph(), output_node_names=OUTPUT_NODE_NAMES,
                              model_name=MODEL_NAME, loss=float(avg_cost))
    proc_data.complete()


if __name__ == "__main__":
    main()
