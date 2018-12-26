# -*- coding: utf-8 -*-
import json
import sys
import collections
from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbColumnProperty,gpudb_ingestor, GPUdbRecord
import time

t1 = time.time()
KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = 'udf_json_example1'

h_db = GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)
tablo = GPUdbTable( None, INPUT_TABLE, db = h_db , use_multihead_ingest = True,multihead_ingest_batch_size=10000, flush_multi_head_ingest_per_insertion=True)
t2 = time.time()
a = json.loads('[{"branchid" : 3,"channelid" : 5,"code" : "MoneyWithdrawed","customerid" : 12345678,"domain" :"Deposits","eventdate" : "2017-08-09","eventtime" : "2017-08-09T17:41:59+03:00","eventtimestamp": 1502289719000,"id": "56541517-ECA1-1563-T014-1B15010AB80C","insertdate": "2017-08-23","inserttime": "2017-08-29T08:04:33.781Z","inserttimestamp": 1503475473781,"payload":"CNzWyxgQjimFEEAAAWUAiA1RSWSoLCOi+rMwFEMD15Q0yEgoGRjAwMDQ0EAEaBgjQzKjMBToBIA==","payloaddecoded":{"customer_id":"12345678","account_suffix_id":5006,"amount":800.0,"currency":"TRY","date":"2017-08-09T14:42:48.028932800Z","acc_reference":{"receipt_no":"F96344","branch_id":1,"date":"2017-08-08T21:00:00Z"},"explanation":""},"subdomain": "APTransactions","topic": "Deposits_APTransactions","topicorder": 19530,"userid": 100006,"version": "8.0"}]')
t3 = time.time()

a = a * 100000
t4 = time.time()
WIDTH = len(a)

#wtype = tablo.get_table_type()
#datum = collections.OrderedDict()
#bulkinserter = gpudb_ingestor.GPUdbIngestor(h_db, INPUT_TABLE,wtype, 10000)
records = []
for i in xrange(0, WIDTH):
   records.append([a[i]["branchid"],a[i]["channelid"],a[i]["code"],a[i]["customerid"],a[i]["domain"],a[i]["eventdate"],
                   a[i]["eventtime"],a[i]["eventtimestamp"],a[i]["id"],a[i]["insertdate"],a[i]["inserttime"],a[i]["inserttimestamp"],a[i]["payload"],
                   a[i]["payloaddecoded"]["account_suffix_id"],a[i]["payloaddecoded"]["amount"],a[i]["payloaddecoded"]["currency"],
                   a[i]["payloaddecoded"]["acc_reference"]["receipt_no"],a[i]["payloaddecoded"]["explanation"],
                   a[i]["subdomain"],a[i]["topic"],a[i]["topicorder"],a[i]["userid"]])
tablo.insert_records(records)

#for i in xrange(0, WIDTH):
#    datum["branchid"] = a[i]["branchid"]
#    datum["channelid"] = a[i]["channelid"]
#    datum["code"] = a[i]["code"]
#    datum["customerid"] = a[i]["customerid"]
#    datum["domain"] = a[i]["domain"]
#    datum["eventdate"] = a[i]["eventdate"]
#    datum["eventtime"] = a[i]["eventtime"]
#    datum["eventtimestamp"] = a[i]["eventtimestamp"]
#    datum["id"] = a[i]["id"]
#    datum["insertdate"] = a[i]["insertdate"]
#    datum["inserttime"] = a[i]["inserttime"]
#    datum["inserttimestamp"] = a[i]["inserttimestamp"]
#    datum["payload"] = a[i]["payload"]
#    datum["account_suffix_id"] = a[i]["payloaddecoded"]["account_suffix_id"]
#    datum["amount"] = a[i]["payloaddecoded"]["amount"]
#    datum["currency"] = a[i]["payloaddecoded"]["currency"]
#    datum["receipt_no"] = a[i]["payloaddecoded"]["acc_reference"]["receipt_no"]
#    datum["explanation"] = a[i]["payloaddecoded"]["explanation"]
#    datum["subdomain"] = a[i]["subdomain"]
#    datum["topic"] = a[i]["topic"]
#    datum["topicorder"] = a[i]["topicorder"]
#    datum["userid"] = a[i]["userid"]
    
t5 = time.time()

#bulkinserter = gpudb_ingestor.GPUdbIngestor(h_db, INPUT_TABLE,wtype, 10000,options = datum)
#kayit = GPUdbRecord(wtype, datum1)
#bulkinserter.insert_record(kayit,record_encoding = 'binary')
#tablo.flush_data_to_server()

#response = bulkinserter.insert_record(kayit,record_encoding = 'binary')
#bulkinserter.flush()
t6 = time.time()
print( 'connection: ' + str(t2-t1) + ' variable declaration: ' + str(t3-t2) + ' coklama : ' + str(t4-t3) + ' json decoding: ' + str(t5-t4) + ' write: ' + str(t6-t1) )

