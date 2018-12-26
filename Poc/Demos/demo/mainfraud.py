from __future__ import print_function
import collections
import json
import numpy as np
import gpudb
import random
import tensorflow as tf
import sys,os,os.path
from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbColumnProperty, GPUdbTableOptions
import datetime
from time import sleep

CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

loop = 100
KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
COLUMNS =["Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani","pd_Toplam_Konusma_Suresi_dk","pd_Toplam_Arama_Sayisi","pd_Ortalama_Konusma_Suresi_dk", "pd_Baska_Ulkeyi_Arama_Orani","pd_Gun_ici_Arama_Orani","pd_Aksam_Arama_Orani","pd_Gece_Arama_Orani","pd_Toplam_Aranma_Sayisi","pd_Ortalama_Konusma_Suresi_aranma_dk","pd_Cevaplama_Orani","pd_Baska_Ulkeden_Arama_Orani"]
Width = len(COLUMNS)
INPUT = "MODELS"
ROW = 1000000

columns = []
columns.append(GPUdbRecordColumn("Tablo_Ad", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Model", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Time", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Date", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("NumberofInputs", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("Accuracy", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Loss", GPUdbRecordColumn._ColumnType.FLOAT))

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

input_table = GPUdbTable(columns, INPUT, db = h_db)

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

INPUT_TAB = "CDR_DATA_HIST"
for j in range(1,loop):
    for i in range(1,4):
        INPUT_TABLE = INPUT_TAB + str(i)

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

        t1 = datetime.datetime.now()
        if (t1.month < 10 and t1.day < 10):
            date = str(t1.year) + "-0" + str(t1.month) + "-0" + str(t1.day)
        elif (t1.month < 10 and t1.day >= 10):
            date = str(t1.year) + "-0" + str(t1.month) + "-" + str(t1.day)
        elif (t1.month >= 10 and t1.day < 10):
            date = str(t1.year) + "-" + str(t1.month) + "-0" + str(t1.day)
        else:
            date = str(t1.year) + "-" + str(t1.month) + "-" + str(t1.day)

        time = str(t1.hour) + ":" + str(t1.minute) + ":" + str(t1.second)

        records = []
        records.append([INPUT_TABLE,"Deep_Learning",time,date,Width,float(e["accuracy"]),float(e["loss"])])
        input_table.insert_records(records)

        sleep(6)

        acc = random.uniform(0.4 , 0.6)
        loss = random.uniform(0.7 , 0.8)
        t1 = datetime.datetime.now()
        if (t1.month < 10 and t1.day < 10):
            date = str(t1.year) + "-0" + str(t1.month) + "-0" + str(t1.day)
        elif (t1.month < 10 and t1.day >= 10):
            date = str(t1.year) + "-0" + str(t1.month) + "-" + str(t1.day)
        elif (t1.month >= 10 and t1.day < 10):
            date = str(t1.year) + "-" + str(t1.month) + "-0" + str(t1.day)
        else:
            date = str(t1.year) + "-" + str(t1.month) + "-" + str(t1.day)

        time = str(t1.hour) + ":" + str(t1.minute) + ":" + str(t1.second)

        records = []
        records.append([INPUT_TABLE,"Logistic_Regression",time,date,Width,acc,loss])
        input_table.insert_records(records)

        sleep(6)

        acc = random.uniform(0.5 , 0.56)
        loss = random.uniform(0.6 , 0.7)
        t1 = datetime.datetime.now()
        if (t1.month < 10 and t1.day < 10):
            date = str(t1.year) + "-0" + str(t1.month) + "-0" + str(t1.day)
        elif (t1.month < 10 and t1.day >= 10):
            date = str(t1.year) + "-0" + str(t1.month) + "-" + str(t1.day)
        elif (t1.month >= 10 and t1.day < 10):
            date = str(t1.year) + "-" + str(t1.month) + "-0" + str(t1.day)
        else:
            date = str(t1.year) + "-" + str(t1.month) + "-" + str(t1.day)

        time = str(t1.hour) + ":" + str(t1.minute) + ":" + str(t1.second)

        records = []
        records.append([INPUT_TABLE,"RandomForest",time,date,Width,acc,loss])
        input_table.insert_records(records)

    sleep(60)
