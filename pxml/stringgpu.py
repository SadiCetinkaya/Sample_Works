from __future__ import print_function

import numpy as np
import tensorflow as tf
import datetime
import sys,os,os.path
# Update PATH with location of nvcc compiler
CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))
#Set XDG_CACHE_HOME directory for PyCUDA to use for temp files;
#will use HOME directory for temp space if not set, which for gpudb_proc
#is /home/gpudb, where it doesn't have write access

log_device_placement = True

c1 = []

#A = tf.constant('asd',dtype = tf.string)
#B = tf.constant('qwe',dtype = tf.string)

A = ['asd','sado']
B = ['qwe','mazo']

def xmlencode(x, y):
    return x+y

with tf.device('/gpu:0'):
    a = tf.placeholder(tf.string)
    b = tf.placeholder(tf.string)
    c1.append(xmlencode(a, b))

with tf.device('/cpu:0'):
    sad = tf.string_join(c1)

t1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement=log_device_placement)) as sess:
    print("deneme %s" % sess.run(sad,feed_dict={a:A, b:B}))
t2 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2-t1))
