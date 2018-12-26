from __future__ import print_function
import collections
import json
import numpy as np
import gpudb
import random
import tensorflow as tf
import sys,os,os.path

CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = "CDR_DATA_HIST1"

COLUMNS =["Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani"]
ROW = 1000000

learning_rate = 0.001
batch_size = 1000
display_step = 1
model_path = "/home/gpudb/fraudcase/results0/model.ckpt"

n_hidden_1 = 128
n_hidden_2 = 128
n_input = 11
n_classes = 1

h_db = gpudb.GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)
InputTrain = h_db.get_records_by_column(table_name = INPUT_TABLE, column_names = COLUMNS, offset = 0,  limit = ROW,encoding = "json")
LabelTrain = h_db.get_records_by_column(table_name = INPUT_TABLE, column_names = "Is_Fraud", offset = 0,  limit = ROW,encoding = "json")
InputTest = h_db.get_records_by_column(table_name = INPUT_TABLE, column_names = COLUMNS, offset = ROW,  limit = ROW/5,encoding = "json")
LabelTest = h_db.get_records_by_column(table_name = INPUT_TABLE, column_names = "Is_Fraud", offset = ROW,  limit = ROW/5,encoding = "json")

Train_Input = np.array(h_db.parse_dynamic_response(InputTrain)['response'].values(),dtype='f')
Train_Label = np.array(h_db.parse_dynamic_response(LabelTrain)['response']['Is_Fraud'],dtype='f')
Test_Input = np.array(h_db.parse_dynamic_response(InputTest)['response'].values(),dtype='f')
Test_Label = np.array(h_db.parse_dynamic_response(LabelTest)['response']['Is_Fraud'],dtype='f')

Train_Input = np.transpose(Train_Input)
Train_Label = np.transpose(Train_Label)
Test_Input = np.transpose(Test_Input)
Test_Label = np.transpose(Test_Label)
Test_Label = np.reshape(Test_Label,(ROW/5,1))


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

print("Starting 1st session...")
with tf.Session() as sess:

    sess.run(init)

    for epoch in range(3):
        avg_cost = 0.
        total_batch = int(ROW/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size, Train_Input, Train_Label)
	    batch_y = np.reshape(batch_y,(batch_size,1))
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("First Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: Test_Input, y: Test_Label}))

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)

    # Resume training
    for epoch in range(7):
        avg_cost = 0.
        total_batch = int(ROW / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size,Train_Input, Train_Label)
	    batch_y = np.reshape(batch_y,(batch_size,1))
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Second Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval(
        {x: Test_Input, y: Test_Label}))
