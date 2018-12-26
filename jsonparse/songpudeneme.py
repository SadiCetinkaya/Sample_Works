# -*- coding: utf-8 -*-
import json
import collections
from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbColumnProperty,gpudb_ingestor, GPUdbRecord
import time
import tensorflow as tf
import sys,os,os.path

# Update PATH with location of nvcc compiler
CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))

t1 = time.time()
KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = 'udf_json_example1'

h_db = GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

tablo = GPUdbTable([['branchid','int'], ['channelid','int'], ['code','string','data'], ['customerid','int'], ['domain','string','data'], ['eventdate','string','date'], ['eventtime','string','data'],['eventtimestamp','long','timestamp'],['id','string','data'],['insertdate','string','date'],['inserttime','string','data'],['inserttimestamp','long','timestamp'],['payload','string','data'],['account_suffix_id','int'],['amount','float'],['currency','string','data'],['receipt_no','string','data'],['explanation','string','data'],['subdomain','string','data'],['topic','string','data'],['topicorder','int'],['userid','int'] ], INPUT_TABLE, db = h_db ,use_multihead_ingest=True)

t2 = time.time()
J = json.loads('[{"branchid" : 3,"channelid" : 5,"code" : "MoneyWithdrawed","customerid" : 12345678,"domain" :"Deposits","eventdate" : "2017-08-09","eventtime" : "2017-08-09T17:41:59+03:00","eventtimestamp": 1502289719000,"id": "56541517-ECA1-1563-T014-1B15010AB80C","insertdate": "2017-08-23","inserttime": "2017-08-29T08:04:33.781Z","inserttimestamp": 1503475473781,"payload":"CNzWyxgQjimFEEAAAWUAiA1RSWSoLCOi+rMwFEMD15Q0yEgoGRjAwMDQ0EAEaBgjQzKjMBToBIA==","payloaddecoded":{"customer_id":"12345678","account_suffix_id":5006,"amount":800.0,"currency":"TRY","date":"2017-08-09T14:42:48.028932800Z","acc_reference":{"receipt_no":"F96344","branch_id":1,"date":"2017-08-08T21:00:00Z"},"explanation":""},"subdomain": "APTransactions","topic": "Deposits_APTransactions","topicorder": 19530,"userid": 100006,"version": "8.0"}]')
t3 = time.time()

J = J * 500000
t4 = time.time()

wtype = tablo.get_table_type()
batch_size=10000
bulkinserter = gpudb_ingestor.GPUdbIngestor(h_db, INPUT_TABLE,wtype, 10000000, options={}, workers=None)

t5 = time.time()
log_device_placement = True
datum = collections.OrderedDict()
kayit = []
def parsing(a):
    datum["branchid"] = a["branchid"]
    datum["channelid"] = a["channelid"]
    datum["code"] = a["code"]
    datum["customerid"] = a["customerid"]
    datum["domain"] = a["domain"]
    datum["eventdate"] = a["eventdate"]
    datum["eventtime"] = a["eventtime"]
    datum["eventtimestamp"] = a["eventtimestamp"]
    datum["id"] = a["id"]
    datum["insertdate"] = a["insertdate"]
    datum["inserttime"] = a["inserttime"]
    datum["inserttimestamp"] = a["inserttimestamp"]
    datum["payload"] = a["payload"]
    datum["account_suffix_id"] = a["payloaddecoded"]["account_suffix_id"]
    datum["amount"] = a["payloaddecoded"]["amount"]
    datum["currency"] = a["payloaddecoded"]["currency"]
    datum["receipt_no"] = a["payloaddecoded"]["acc_reference"]["receipt_no"]
    datum["explanation"] = a["payloaddecoded"]["explanation"]
    datum["subdomain"] = a["subdomain"]
    datum["topic"] = a["topic"]
    datum["topicorder"] = a["topicorder"]
    datum["userid"] = a["userid"]
    return datum    

with tf.device('/gpu:0'):
    j = tf.placeholder(tf.string)
    kayit.append(parsing(j))

with tf.device('/cpu:0'):
    sad = tf.string_join(c1,separator = '')

with tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement=log_device_placement)) as sess:
    asd = sess.run(sad,feed_dict={j:J})

for i in xrange(0, WIDTH):
    bulkinserter.insert_record(kayit,record_encoding = 'binary')
# an explicit flush to help make the insert time consistently measurable
bulkinserter.flush()

t6 = time.time()
print( 'connection: ' + str(t2-t1) + ' variable declaration: ' + str(t3-t2) + ' coklama : ' + str(t4-t3) + ' json decoding: ' + str(t5-t4) +' full load: ' + str(t6-t5) )

