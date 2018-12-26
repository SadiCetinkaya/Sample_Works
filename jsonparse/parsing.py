import json
from json import loads
import sys
import collections
from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbColumnProperty

KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = 'udf_json_example'

h_db = GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

a = json.loads('[{"RecId" : 1,"Year" : 1985,"TeamId" : "GS","LeagueId" : "TR","PlayerId" :"barkele01","Salary" : 870000},{"RecId" : 2,"Year" : 1985,"TeamId" : "PK","LeagueId" : "PL","PlayerId" : "bedrost01","Salary" : 550000}]')
WIDTH = len(a)

columns = []
columns.append(GPUdbRecordColumn("RecId", GPUdbRecordColumn._ColumnType.INT, [GPUdbColumnProperty.PRIMARY_KEY, GPUdbColumnProperty.INT16]))
columns.append(GPUdbRecordColumn("Year", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("TeamId", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("LeagueId", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("PlayerId", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Salary", GPUdbRecordColumn._ColumnType.INT))

if h_db.has_table(table_name = INPUT_TABLE)['table_exists']:
   h_db.clear_table(table_name = INPUT_TABLE)
input_table = GPUdbTable(columns, INPUT_TABLE, db = h_db)

records = []
for i in range(0, WIDTH):
   records.append([a[i]["RecId"],a[i]["Year"],a[i]["TeamId"],a[i]["LeagueId"],a[i]["PlayerId"],a[i]["Salary"]])
input_table.insert_records(records)

