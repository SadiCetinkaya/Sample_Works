import sys
import gpudb

KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
proc_name = 'c3'
file_name = proc_name + '.py'
INPUT_TABLE = 'udf_cublas_in_table'

# Read proc code in as bytes and add to a file data array
files = {}
with open(file_name, 'rb') as file:
    files[file_name] = file.read()

# Connect to Kinetica
h_db = gpudb.GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

# Remove proc if it exists from a prior registration
if h_db.has_proc(proc_name)['proc_exists']:
    h_db.delete_proc(proc_name)

print "Registering proc..."
response = h_db.create_proc(proc_name, 'distributed', files, 'python', [file_name], {})
print response

print "Executing proc..."
response = h_db.execute_proc(proc_name, {}, {}, [INPUT_TABLE], {}, [], {})
print response

