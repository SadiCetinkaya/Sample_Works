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

COLUMNS =["Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani","pd_Toplam_Konusma_Suresi_dk","pd_Toplam_Arama_Sayisi","pd_Ortalama_Konusma_Suresi_dk", "pd_Baska_Ulkeyi_Arama_Orani","pd_Gun_ici_Arama_Orani","pd_Aksam_Arama_Orani","pd_Gece_Arama_Orani","pd_Toplam_Aranma_Sayisi","pd_Ortalama_Konusma_Suresi_aranma_dk","pd_Cevaplama_Orani","pd_Baska_Ulkeden_Arama_Orani"]
ROW = 2600000

learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

n_hidden_1 = 128
n_hidden_2 = 128
num_input = len(COLUMNS)
num_classes = 2
model_path = "/home/gpudb/fraudcase/resultsHIST2"

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

def neural_net(x_dict):
    x = x_dict['input']
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

def model_fn(features, labels, mode):
    logits = neural_net(features)
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

model = tf.estimator.Estimator(model_fn,model_dir=model_path)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'input': Train_Input}, y=Train_Label,
    batch_size=batch_size, num_epochs=None, shuffle=True)

model.train(input_fn, steps=num_steps)
print("train completed")
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'input': Test_Input}, y=Test_Label,
    batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn)

print("Testing Accuracy:", e)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'input': Test_Input},batch_size=batch_size, shuffle=False)

ab = model.predict(Test_Input)

print(ab)

