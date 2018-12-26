from __future__ import print_function
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

import collections
import json
import numpy as np
import gpudb
import random
import tensorflow as tf
import sys,os,os.path

KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = "CDR_DATA_HIST"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

COLUMNS =["Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani"]
ROW = 100000

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

Train_Input = np.reshape(Train_Input,(ROW,11))
Train_Label = np.reshape(Train_Label,(ROW,1))
Test_Input = np.reshape(Test_Input,(ROW/5,11))
Test_Label = np.reshape(Test_Label,(ROW/5,1))

num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 1 # The 10 digits
num_features = 11 # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

saver = tf.train.Saver()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init_vars)

    for i in range(1, num_steps + 1):

        batch_x, batch_y = next_batch(batch_size, Train_Input, Train_Label)
        _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

    print("Optimization Finished!")
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    test_x, test_y = Test_Input, Test_Label
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
