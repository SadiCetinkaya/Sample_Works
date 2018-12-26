import sys
import collections
from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbColumnProperty

KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = 'udf_sos_in_table'
OUTPUT_TABLE = 'udf_sos_out_table'
MAX_RECORDS = 10000

## Connect to Kinetica
h_db = GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

## Create input data table
columns = []
columns.append(GPUdbRecordColumn("id", GPUdbRecordColumn._ColumnType.INT, [GPUdbColumnProperty.PRIMARY_KEY, GPUdbColumnProperty.INT16]))
columns.append(GPUdbRecordColumn("x1", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("x2", GPUdbRecordColumn._ColumnType.FLOAT))

if h_db.has_table(table_name = INPUT_TABLE)['table_exists']:
   h_db.clear_table(table_name = INPUT_TABLE)
input_table = GPUdbTable(columns, INPUT_TABLE, db = h_db)

## Insert input data
import random

records = []
for val in range(1, MAX_RECORDS+1):
   records.append([val, random.gauss(1,1), random.gauss(1,2)])
input_table.insert_records(records)


## Create output data table
columns = []
columns.append(GPUdbRecordColumn("id", GPUdbRecordColumn._ColumnType.INT, [GPUdbColumnProperty.PRIMARY_KEY, GPUdbColumnProperty.INT16]))
columns.append(GPUdbRecordColumn("y", GPUdbRecordColumn._ColumnType.FLOAT))

if h_db.has_table(table_name = OUTPUT_TABLE)['table_exists']:
   h_db.clear_table(table_name = OUTPUT_TABLE)
GPUdbTable(columns, OUTPUT_TABLE, db = h_db)


