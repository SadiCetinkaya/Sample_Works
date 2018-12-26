from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import gpudb
import collections
import DatabaseDefinitions as db


MINST_URL = "MNIST_data/"


def main():
    logging_with_warnings = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = input_data.read_data_sets(MINST_URL, one_hot=True)
    db_handle = gpudb.GPUdb(encoding='BINARY', host=db.HOST_IP, port=db.PORT)
    print("creating tables in database")
    setup_tables(db_handle, db.TRAIN_INPUT_TABLE, db.INFERENCE_INPUT_TABLE, db.TRAIN_INFERENCE_INPUT_TYPE,
                 db.INFERENCE_OUTPUT_TABLE, db.INFERENCE_OUTPUT_TYPE, db.MODEL_TABLE, db.MODEL_TYPE,
                 db.MODEL_TYPE_PROPERTIES)
    # ingest training data
    print("start loading train data into table")
    ingest(db_handle, db.TRAIN_INPUT_TABLE, mnist.train, db.TRAIN_INFERENCE_INPUT_TYPE)
    # ingest inference data
    print("start loading inference input data (test data) into table")
    ingest(db_handle, db.INFERENCE_INPUT_TABLE, mnist.test, db.TRAIN_INFERENCE_INPUT_TYPE)
    tf.logging.set_verbosity(logging_with_warnings)


def setup_tables(db_handle, train_input_table, inference_input_table, train_inference_input_type,
                 inference_output_table, inference_output_type, model_table, model_table_type, model_type_properties):
    # create input training data table
    response = db_handle.create_type(type_definition=train_inference_input_type, label=train_input_table + '_lbl',
                                     properties={})
    if db_handle.has_table(table_name=train_input_table)['table_exists']:
        db_handle.clear_table(table_name=train_input_table)
    response = db_handle.create_table(table_name=train_input_table, type_id=response['type_id'],
                                      options={"collection_name": db.COLLECTION})
    print("Create train_input_table response status: {}".format(response['status_info']['status']))
    # create input inference data table
    response = db_handle.create_type(type_definition=train_inference_input_type, label=inference_input_table + '_lbl',
                                     properties={})
    if db_handle.has_table(table_name=inference_input_table)['table_exists']:
        db_handle.clear_table(table_name=inference_input_table)
    response = db_handle.create_table(table_name=inference_input_table, type_id=response['type_id'],
                                      options={"collection_name": db.COLLECTION})
    print("Create inference_input_table response status: {}".format(response['status_info']['status']))
    # create inference output table
    response = db_handle.create_type(type_definition=inference_output_type, label=inference_output_table + '_lbl',
                                     properties=db.INFERENCE_OUTPUT_TYPE_PROPERTIES)
    if db_handle.has_table(table_name=inference_output_table)['table_exists']:
        db_handle.clear_table(table_name=inference_output_table)
    response = db_handle.create_table(table_name=inference_output_table, type_id=response['type_id'],
                                      options={"collection_name": db.COLLECTION})
    print("Create inference_output_table response status: {}".format(response['status_info']['status']))
    # create table for storing models
    if not db_handle.has_table(table_name=model_table)['table_exists']:
        response = db_handle.create_type(type_definition=model_table_type, label=model_table,
                                         properties=model_type_properties)
        response = db_handle.create_table(table_name=model_table, type_id=response['type_id'],
                                          options={"collection_name": db.COLLECTION})
        print("Create model_table response status: {}".format(response['status_info']['status']))


def ingest(db_handle, target_table, dataset, table_type, start_id=0):
    i = start_id
    encoded_object_list = []
    for data, label in zip(dataset.images, dataset.labels):
        datum = collections.OrderedDict()
        datum["id"], datum["data"], datum["onehot_label"], datum["readable_label"] = \
            int(i), data.tobytes(), label.tobytes(), int(np.argwhere(label == 1)[0][0])
        encoded_object_list.append(db_handle.encode_datum(table_type, datum))
        i = i + 1
        if i % db.DATA_PACK_SIZE == 0:
            response = db_handle.insert_records(table_name=target_table, data=encoded_object_list,
                                                list_encoding='binary', options={})
            if response['status_info']['status'] == "ERROR":
                print("Insert into {} response status: {}".format(target_table, response['status_info']))
            encoded_object_list = []
            pct = float(i)/float(dataset.images.shape[0])*100.0
            if pct == int(pct) and pct % 20 == 0:
                print(str(pct)+" % loaded")
    print("%s table has %d records inserted" % (target_table, i))


if __name__ == '__main__':
    main()
