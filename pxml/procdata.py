from __future__ import print_function
import collections
import json
import random
import string
import numpy as np
import gpudb
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree import ElementTree
from xml.dom import minidom
import datetime
import multiprocessing as mp
import tensorflow as tf
import sys,os,os.path
from kinetica_proc import ProcData

# Update PATH with location of nvcc compiler
CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

#row = 5000000
#row = 64

t1 = datetime.datetime.now()
proc_data = ProcData()
in_table = proc_data.input_data[0]

#kinetica data read
#KINETICA_HOST = '10.20.10.228'
#KINETICA_PORT = '9191'
#h_db = gpudb.GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)
#domain = h_db.get_records_by_column(table_name = "udf_json_example",column_names = ["domain","code","eventtime","id"], offset = 0,  limit = row,encoding = "json")
#domain = h_db.get_records_by_column(table_name = "udf_json_example",column_names = ["domain"], offset = 0,  limit = row,encoding = "json")
#code = h_db.get_records_by_column(table_name = "udf_json_example",column_names = ["code"], offset = 0,  limit = row,encoding = "json")
#eventtime = h_db.get_records_by_column(table_name = "udf_json_example",column_names = ["eventtime"], offset = 0,  limit = row,encoding = "json")
#id1 = h_db.get_records_by_column(table_name = "udf_json_example",column_names = ["id"], offset = 0,  limit = row,encoding = "json")

t2 = datetime.datetime.now()

t3 = datetime.datetime.now()
#A1 = h_db.parse_dynamic_response(domain)['response']['domain']
#B1 = h_db.parse_dynamic_response(domain)['response']['code']
#C1 = h_db.parse_dynamic_response(domain)['response']['eventtime']
#D1 = h_db.parse_dynamic_response(domain)['response']['id']
#t4 = datetime.datetime.now()

#tablo = gpudb.GPUdbRecord.decode_binary_data(response["type_schema"], response["records_binary"])
t5 = datetime.datetime.now()
A1 = np.ndarray(shape=(in_table.size, 1), dtype=str).astype(np.string_)
B1 = np.ndarray(shape=(in_table.size, 1), dtype=str).astype(np.string_)

for i in xrange(0, in_table.size):
    A1[i,0] = in_table['domain'][i]
    B1[i,0] = in_table['code'][i]

#A = A1[:(row/4)]
#B = B1[:(row/4)]
#C = A1[(row/4):(row/2)]
#D = B1[(row/4):(row/2)]
#E = A1[(row/2):(3*row/4)]
#F = B1[(row/2):(3*row/4)]
#G = A1[(3*row/4):]
#H = B1[(3*row/4):]
t6 = datetime.datetime.now()


t7 = datetime.datetime.now()
log_device_placement = True

c1 = []

def retrieve(x,y):
    return '<record recordtype="E"><domain>'+ x + '</domain><code>' + y + '</code></record>'

with tf.device('/gpu:0'):
    a = tf.placeholder(tf.string)
    b = tf.placeholder(tf.string)
    c1.append(retrieve(a,b))

#with tf.device('/gpu:1'):
#    c = tf.placeholder(tf.string)
#    d = tf.placeholder(tf.string)
#    c1.append(retrieve(c,d))

#with tf.device('/gpu:2'):
#    e = tf.placeholder(tf.string)
#    f = tf.placeholder(tf.string)
#    c1.append(retrieve(e,f))

#with tf.device('/gpu:3'):
#    g = tf.placeholder(tf.string)
#    h = tf.placeholder(tf.string)
#    c1.append(retrieve(g,h))

with tf.device('/cpu:0'):
    sad = tf.string_join(c1,separator = '')

#t3 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement=log_device_placement)) as sess:
    #print("deneme %s" % sess.run(sad,feed_dict={a:A,b:B,c:C,d:D,e:E,f:F,g:G,h:H}))
    asd = sess.run(sad,feed_dict={a:A,b:B}) 
    #print(asd)
t8 = datetime.datetime.now()
print("connection and retrieve: " + str(t2-t1) + " parse: " + str(t4-t3) + " cpu assign: " + str(t6-t5) + " gpu process: " + str(t8-t7) + " TOTAL: " + str(t8-t1))
proc_data.complete()
