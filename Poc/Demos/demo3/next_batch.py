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
INPUT_TABLE = "CDR_DATA_FRAUD_1"

COLUMNS =["Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani"]
ROW = 20

learning_rate = 0.01
batch_size = 100
display_step = 1
model_path = "/home/gpudb/fraudcase/model.ckpt"

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 11
n_classes = 2

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
Train_Label =np.transpose(Train_Label)
Test_Input = np.transpose(Test_Input)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

print(Train_Input)
print(Train_Label)

Xtr, Ytr = next_batch(5, Train_Input, Train_Label)
print('\n5 random samples')
print(Xtr)
print(Ytr)

