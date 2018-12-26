import tensorflow as tf
import numpy as np
import sys,os,os.path
import datetime
import multiprocessing as mp
from numpy import genfromtxt

CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH']+os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

row = 1000 * 1000 * 200
vek = 128
batch = 1000 * 1000 * 10
log_device_placement = True

t1 = datetime.datetime.now()
x = np.asarray(np.random.rand(1,vek),np.float32)
y = np.asarray(np.random.rand(row,vek),np.float32)
y1 = y[:row/2]
y2 = y[row/2:]
t2 = datetime.datetime.now()

a = tf.placeholder(tf.float32,[1, vek]) 
# b = tf.placeholder(tf.float32,[row,vek])
# sonuc_gpu = tf.linalg.norm(a-b,axis=1)
log_device_placement = True
t3 = datetime.datetime.now()

#if (row>batch):
#   turn = int(round(row/batch))
#   b = tf.placeholder(tf.float32,[batch,vek])
#else:
#   turn = 1
#   b = tf.placeholder(tf.float32,[row,vek])

#sonuc_gpu = tf.linalg.norm(a-b,axis=1)


if (y1.shape[0]>batch):
   turn = int(round(y1.shape[0]/batch))
else:
   turn = 1

c1 =[]
c2 = []
with tf.device('gpu:0'):
   
   a = tf.placeholder(tf.float32,[1, vek])

   if (y1.shape[0]>batch):
      b = tf.placeholder(tf.float32,[batch,vek])
   else:
      b = tf.placeholder(tf.float32,[y1.shape[0],vek])
   c1.append(tf.linalg.norm(a-b,axis=1))
 
with tf.device('gpu:1'):

#   a = tf.placeholder(tf.float32,[1, vek])

   if (y1.shape[0]>batch):
      c = tf.placeholder(tf.float32,[batch,vek])
   else:
      c = tf.placeholder(tf.float32,[y2.shape[0],vek])
   c2.append(tf.linalg.norm(a-b,axis=1))


with tf.device('cpu:0'):
   sonuc_gpu = tf.concat([c1,c2],0)

for i in range(0,turn):
   with tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement=log_device_placement)) as sess:
      asd = sess.run(sonuc_gpu,feed_dict={a:x, b:y1[i*batch:(i+1)*batch,],c:y2[i*batch:(i+1)*batch,]})
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
