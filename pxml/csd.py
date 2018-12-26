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
# Update PATH with location of nvcc compiler
CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

#row = 5000000
row = 64

t1 = datetime.datetime.now()
#kinetica data read
KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
h_db = gpudb.GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)
table1 = gpudb.GPUdbTable( None, 'udf_json_example', db = h_db )
A = table1.get_records_by_column('domain')
B = table1.get_records_by_column('code') 
C = table1.get_records_by_column('eventtime')
D = table1.get_records_by_column('id')

t2 = datetime.datetime.now()

for col, vals in A.items():
    A1 = "%s" % col,vals
for col, vals in B.items():
    B1 = "%s" % col,vals
for col, vals in C.items():
    C1 = "%s" % col,vals
for col, vals in D.items():
    D1 = "%s" % col,vals

t3 = datetime.datetime.now()

#tablo = gpudb.GPUdbRecord.decode_binary_data(response["type_schema"], response["records_binary"])

#A = A1[:(row/4)]
#B = B1[:(row/4)]
#C = A1[(row/4):(row/2)]
#D = B1[(row/4):(row/2)]
#E = A1[(row/2):(3*row/4)]
#F = B1[(row/2):(3*row/4)]
#G = A1[(3*row/4):]
#H = B1[(3*row/4):]
t4 = datetime.datetime.now()

log_device_placement = True

c1 = []

def retrieve(x,y,w,z):
    return '<record recordtype="E"><domain>'+ x + '</domain><code>' + y + '</code></record>' + '<eventtime>' + w + '</eventtime><branchid>' + z + '</branchid>' + "\n"

B = table1.get_records_by_column('code')
#for gp in ['/gpu:0','/gpu:1','/gpu:2','/gpu:3']:
with tf.device('gpu:0'):
    a = tf.placeholder(tf.string)
    b = tf.placeholder(tf.string)
    c = tf.placeholder(tf.string)
    d = tf.placeholder(tf.string)
    c1.append(retrieve(a,b,c,d))

with tf.device('/cpu:0'):
    sad = tf.string_join(c1,separator = '')


with tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement=log_device_placement)) as sess:
    #print("deneme %s" % sess.run(sad,feed_dict={a:A,b:B,c:C,d:D,e:E,f:F,g:G,h:H}))
    asd = sess.run(sad,feed_dict={a:A1,b:B1,c:C1,d:D1}) 
    print(asd)
t5 = datetime.datetime.now()
print("connection and retrieve: " + str(t2-t1) + " parse: " + str(t3-t2) + " cpu assign: " + str(t4-t3) + " gpu process: " + str(t5-t4) + " TOTAL: " + str(t5-t1))
