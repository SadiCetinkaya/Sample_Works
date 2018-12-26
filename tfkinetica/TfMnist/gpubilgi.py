from tensorflow.python.client import device_lib
import os

CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

# System parameter setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(device_lib.list_local_devices())

