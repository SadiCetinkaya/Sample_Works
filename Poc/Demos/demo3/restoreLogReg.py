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
INPUT_TABLE = "CDR_DATA_HIST"

COLUMNS =["Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani"]
ROW = 50

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



learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
model_path = "/home/gpudb/fraudcase/results2/model.ckpt"

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

x = tf.placeholder(tf.float32, [None, 11]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 1]) # 0-9 digits recognition => 10 classes

W = tf.Variable(tf.zeros([11, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
prediction = tf.argmax(pred,1)
# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % model_path)
    # Training cycle
    for i in range(1,ROW):
	#prediction_run = sess.run(prediction, feed_dict={x:np.reshape(Train_Input[i],(1,11))})
	print(Train_Label[i], "predictions", float(prediction.eval(feed_dict={x:np.reshape(Train_Input[i],(1,11))})))
	#print("Orjinal: ", Train_Label[i], "Prediction: ", float(prediction_run))
print(np.histogram(Train_Input[:,0]))
