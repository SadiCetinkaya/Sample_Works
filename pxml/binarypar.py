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
    global i 
    #row = 5000000
    row = 8
    KINETICA_HOST = '10.20.10.228'
    KINETICA_PORT = '9191'

    h_db = gpudb.GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)
    table1 = gpudb.GPUdbTable( None, 'udf_json_example', db = h_db )
    records = table1.get_records_by_column('domain')
    #response = h_db.get_records_by_column(table_name = "udf_json_example",column_names = ["domain","branchid"], offset = 0,  limit = row,encoding = "binary")
    #d = h_db.parse_dynamic_response(response)['response']['domain']
    #e = h_db.parse_dynamic_response(response)['response']['branchid']
    #print(len(records))
    for col, vals in records.items():
        print ("Row number: %s" % col, len(vals))
if __name__ == '__main__':
   a = time.time()
   retrieve()
   b = time.time()
   print('GPUdbTable data read speed: ' + str(b-a))

