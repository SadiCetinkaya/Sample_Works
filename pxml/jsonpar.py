from __future__ import print_function

import collections
from collections import OrderedDict
import json
import numpy as np
import gpudb
import time
import sys
import cStringIO
from avro import schema, io

def retrieve():
    KINETICA_HOST = '10.20.10.228'
    KINETICA_PORT = '9191'

    h_db = gpudb.GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)
    response = h_db.get_records_by_column(table_name = "udf_json_example",column_names = ["domain"], offset = 0,  limit = 10000,encoding = "json")
    d = h_db.parse_dynamic_response(response)['response']['domain']
    #e = h_db.parse_dynamic_response(response)['response']['branchid']
    #print(d[1:3])
    print("Row number: " + str(len(d)))
if __name__ == '__main__':
   a = time.time()
   retrieve()
   b = time.time()
   print("json data read speed: " + str(b-a))

