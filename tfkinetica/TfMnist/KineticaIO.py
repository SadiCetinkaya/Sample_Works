import gpudb
import collections
import json
import DatabaseDefinitions as db
import time
from datetime import datetime


class KineticaIO(object):
    def __init__(self, db_handle=gpudb.GPUdb(encoding='BINARY', host=db.HOST_IP, port=db.PORT)):
        self.db_handle = db_handle

    def model_to_kinetica(self, pb_file=None, sess=None, graph=None, output_node_names=None, model_name="Model",
                          loss=0.99):
        import uuid
        from time import gmtime, strftime
        datetime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        from tensorflow import graph_util
        db_handle = self.db_handle
        # generate output binary string
        # output_node_names example,output_node_names = "input,output,output2"
        if pb_file != None:
            model = open(pb_file, 'rb').read()
        else:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                graph.as_graph_def(),
                output_node_names.split(",")  # The output node names are used to select the useful nodes
            )
            model = output_graph_def.SerializeToString()
        # insert model into kinetica
        encoded_obj_list = []
        model_id = str(uuid.uuid1())
        datum = collections.OrderedDict()
        datum["binary"] = model
        datum["name"] = model_name
        datum["id"] = model_id
        datum["loss"] = loss
        datum["date_time_created"] = datetime
        encoded_obj_list.append(db_handle.encode_datum(db.MODEL_TYPE, datum))
        options = {'update_on_existing_pk': 'true'}
        response = db_handle.insert_records(table_name=db.MODEL_TABLE, data=encoded_obj_list,
                                            list_encoding='binary', options=options)
        print(response)

    def model_from_kinetica(self, model_id):
        from tensorflow import GraphDef, Graph, import_graph_def
        h_db = self.db_handle
        response = h_db.get_records(table_name=db.MODEL_TABLE, encoding="binary",
                                    options={'expression': "id=\"" + model_id + "\""})
        records = gpudb.GPUdbRecord.decode_binary_data(response["type_schema"], response["records_binary"])
        record = records[0]
        graph_def = GraphDef()
        graph_def.ParseFromString(record["binary"])
        graph = Graph()
        with graph.as_default():
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            import_graph_def(graph_def)
        return graph

    def get_data(self, table=db.TRAIN_INPUT_TABLE, offset=0, number_data=1):
        h_db = self.db_handle
        response = h_db.get_records(table_name=table, offset=offset, limit=number_data)
        res_decoded = gpudb.GPUdbRecord.decode_binary_data(response["type_schema"], response["records_binary"])
        return res_decoded

    def register_udf(self, proc_name, distributed_mode, file_list, sysargs):
        db_handle = self.db_handle
        files = {}
        for current_file in file_list:
            with open(current_file, 'rb') as f:
                files[current_file] = f.read()
        print db_handle.has_proc(proc_name)
        if db_handle.has_proc(proc_name)['proc_exists']:
            db_handle.delete_proc(proc_name)  # Remove proc if it exists from a prior registration
        print ("Registering udf: " + proc_name)
        response = db_handle.create_proc(proc_name, distributed_mode, files, 'python', sysargs, {})
        print response

    def execute_udf(self, proc_name, input_table_names=[], output_table_names=[]):
        db_handle = self.db_handle
        print "Executing udf..."
        response = db_handle.execute_proc(proc_name, {}, {}, input_table_names, {}, output_table_names, {})
        if response['status_info']['status'] == 'OK':
            run_id = response['run_id']
            print('Proc was launched successfully with run_id: ' + run_id)
            start_time = datetime.now()
            while db_handle.show_proc_status(run_id)['overall_statuses'][run_id] == 'running':
                time.sleep(1)
            final_proc_state = db_handle.show_proc_status(run_id)['overall_statuses'][run_id]
            print("total running time is ", datetime.now() - start_time)
            print('Final Proc state: ' + final_proc_state)
            if final_proc_state == 'error':
                raise RuntimeError('proc error')
        else:
            print('Error launching proc; response: ')
            raise RuntimeError('proc error')

    def get_model_id(self):  # use most recent model from table
        db_handle = self.db_handle
        response = db_handle.get_records_by_column(table_name=db.MODEL_TABLE, offset=0, column_names=['id'],
                                                   limit=1, encoding='json',
                                                   options={"sort_by": "date_time_created", "sort_order": "descending"})
        my_dict = json.loads(response.json_encoded_response)
        model_id = my_dict["column_1"][0]
        return model_id
