HOST_IP = "10.20.10.228"
DATA_PACK_SIZE = 100
PORT = "9191"
COLLECTION = "Mnist"
TRAIN_INPUT_TABLE = "Mnist_training_input"
INFERENCE_INPUT_TABLE = "Mnist_inference_input"
INFERENCE_OUTPUT_TABLE = "Mnist_inference_output"
MODEL_TABLE = "Mnist_train_output"

# train- and inference data input type:
TRAIN_INFERENCE_INPUT_TYPE = """
    {
       "type": "record",
       "name": "input_type",
       "fields": [
          {"name":"id","type":"int"},
          {"name":"data","type":"bytes"},
          {"name":"onehot_label","type":"bytes"},
          {"name":"readable_label","type":"int"}
       ]
    }  """

# table for storing models:
MODEL_TYPE = """
    {
        "type": "record",
        "name": "file_type",
        "fields": [
            {"name":"binary","type":"bytes"},
            {"name":"name","type":"string"},
            {"name":"id","type":"string"},
            {"name":"loss","type":"double"},
            {"name":"date_time_created","type":"string"}
        ]
    }"""

MODEL_TYPE_PROPERTIES = {"id": ["char64"], "date_time_created": ["datetime"]}

# output data type:
INFERENCE_OUTPUT_TYPE = """
    {
       "type": "record",
       "name": "output_type",
       "fields": [
          {"name":"id","type":"int"},
          {"name":"predict","type":"int"},
          {"name":"label","type":"int"},
          {"name":"model_id","type":"string"}
       ]
    }  """

INFERENCE_OUTPUT_TYPE_PROPERTIES = {"model_id": ["char64"]}
