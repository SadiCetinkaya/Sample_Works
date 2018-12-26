import tensorflow as tf
import numpy as np
import sys,os,os.path
import datetime
import multiprocessing as mp
from numpy import genfromtxt

CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH']+os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

row = 10000000
vek = 128
batch = 10000000
log_device_placement = True


t1 = datetime.datetime.now()
x = np.asarray(np.random.rand(1,vek),np.float32)
y = np.asarray(np.random.rand(row,vek),np.float32)
t2 = datetime.datetime.now()

a = tf.placeholder(tf.float32,[1, vek]) 
# b = tf.placeholder(tf.float32,[row,vek])
# sonuc_gpu = tf.linalg.norm(a-b,axis=1)
log_device_placement = True
t3 = datetime.datetime.now()

if (row>batch):
   turn = int(round(row/batch))
   b = tf.placeholder(tf.float32,[batch,vek])
else:
   turn = 1
   b = tf.placeholder(tf.float32,[row,vek])

sonuc_gpu = tf.linalg.norm(a-b,axis=1)

for i in range(0,turn):
   with tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement=log_device_placement)) as sess:
      asd = sess.run(sonuc_gpu,feed_dict={a:x, b:y[i*batch:(i+1)*batch,]})
#   print(asd)
t4 = datetime.datetime.now()

print('Data uretme: ', str(t2-t1))
print('GPU Islem: ', str(t4-t3))

def anakod():
   t5 = datetime.datetime.now()
   sonuc_cpu = np.linalg.norm(y-x,axis=1)
   t6 = datetime.datetime.now()
   print('CPU Islem: ', str(t6-t5))

if __name__ == "__main__":
   for i in range(1,2):
       mp.Process(target = anakod).start()
#   anakod()
