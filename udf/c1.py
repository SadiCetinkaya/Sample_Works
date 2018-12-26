import sys
import collections
from gpudb import GPUdb, GPUdbTable, GPUdbTableOptions, GPUdbRecordColumn
import numpy
KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = 'udf_cublas_in_table'

## Connect to Kinetica
h_db = GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

## Create input data table
columns = []
columns.append(GPUdbRecordColumn("x", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("y", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("z", GPUdbRecordColumn._ColumnType.FLOAT))

if h_db.has_table(table_name = INPUT_TABLE)['table_exists']:
   h_db.clear_table(table_name = INPUT_TABLE)
input_table = GPUdbTable(columns, INPUT_TABLE, db = h_db, options = GPUdbTableOptions.default().is_replicated(True))

## Insert input data
import random

records = []
for val in range(10):
   records.append([random.uniform(0,10), random.uniform(0,10), random.uniform(0,10)])
input_table.insert_records(records)
