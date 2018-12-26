from __future__ import print_function
import tensorflow as tf
import KineticaIO
import struct
import numpy as np
from kinetica_proc import ProcData
import os
import sys
from tensorflow.python.client import device_lib

CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

# System parameter setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(device_lib.list_local_devices())

# If you head node internal and external IPs are different, then use internal IP
gpu_memory_fraction = float(sys.argv[1])
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)


def main():
    # kinetica code to get input table, get model from database to tensorflow graph
    proc_data = ProcData()
    in_table, out_table = proc_data.input_data[0], proc_data.output_data[0]
    out_table.size = in_table.size
    # columns of output table:
    y_id, y_predicted, y_label, y_model_id = out_table[0], out_table[1], out_table[2], out_table[3]
    data_list = in_table["data"]
    kio = KineticaIO.KineticaIO()
    model_id_from_db = kio.get_model_id()
    model_id = struct.Struct(str(len(model_id_from_db))+'s').pack(model_id_from_db.encode('utf-8'))
    print("the model id is ", model_id)
    graph = kio.model_from_kinetica(model_id)
    # tensorflow code, run a session on tensorflow using pre-trained model for inference
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_prediction = graph.get_tensor_by_name("import/output:0")
        tf_train_dataset = graph.get_tensor_by_name("import/input:0")
        data = np.array([np.fromstring(di, dtype=np.float32) for di in data_list])
        predict = sess.run([train_prediction], feed_dict={tf_train_dataset: data})
        print("the inference raw output is ", predict)
        y_predicted[:] = [(np.argmax(record)) for record in predict[0]]
    y_id[:] = in_table["id"]
    y_label[:] = in_table["readable_label"]
    y_model_id[:] = np.repeat(model_id, len(y_id))
    proc_data.complete()


if __name__ == "__main__":
    main()
