from __future__ import print_function
import collections
import json
import numpy as np
import gpudb
import random
import tensorflow as tf
import sys,os,os.path
from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbColumnProperty, GPUdbTableOptions, gpudb_ingestor, GPUdbRecord
import datetime
from time import sleep

loop = 2
KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = "PREDICTIONS"
INPUT = "CDR_DATA_HIST3"

h_db = GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

COLUMNS =["AboneNo","Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani","pd_Toplam_Konusma_Suresi_dk","pd_Toplam_Arama_Sayisi","pd_Ortalama_Konusma_Suresi_dk", "pd_Baska_Ulkeyi_Arama_Orani","pd_Gun_ici_Arama_Orani","pd_Aksam_Arama_Orani","pd_Gece_Arama_Orani","pd_Toplam_Aranma_Sayisi","pd_Ortalama_Konusma_Suresi_aranma_dk","pd_Cevaplama_Orani","pd_Baska_Ulkeden_Arama_Orani","Is_Fraud"]

ROW = 15000

InputTrain = h_db.get_records_by_column(table_name = INPUT, column_names = COLUMNS, offset = 0,  limit = ROW,encoding = "json")
Data = np.array(h_db.parse_dynamic_response(InputTrain)['response'].values())
Data = np.transpose(Data)

columns = []
columns.append(GPUdbRecordColumn("AboneNo", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Tarih", GPUdbRecordColumn._ColumnType.STRING,[GPUdbColumnProperty.DATE]))
columns.append(GPUdbRecordColumn("Saat", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Toplam_Konusma_Suresi_dk", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("Toplam_Arama_Sayisi", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("Ortalama_Konusma_Suresi_dk", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("Baska_Ulkeyi_Arama_Orani", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("Gun_ici_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Aksam_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Gece_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Toplam_Aranma_Sayisi", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("Ortalama_Konusma_Suresi_Aranma_dk", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("Cevaplama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Baska_Ulkeden_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("pd_Toplam_Konusma_Suresi_dk", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("pd_Toplam_Arama_Sayisi", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("pd_Ortalama_Konusma_Suresi_dk", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("pd_Baska_Ulkeyi_Arama_Orani", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("pd_Gun_ici_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("pd_Aksam_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("pd_Gece_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("pd_Toplam_Aranma_Sayisi", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("pd_Ortalama_Konusma_Suresi_aranma_dk", GPUdbRecordColumn._ColumnType.INT))
columns.append(GPUdbRecordColumn("pd_Cevaplama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("pd_Baska_Ulkeden_Arama_Orani", GPUdbRecordColumn._ColumnType.FLOAT))

input_table = GPUdbTable(columns, INPUT_TABLE, db = h_db)

print(Data[50:70,0])

#:for j in range(1,loop):
t1 = datetime.datetime.now()
if (t1.month < 10 and t1.day < 10):
   date = str(t1.year) + "-0" + str(t1.month) + "-0" + str(t1.day)
elif (t1.month < 10 and t1.day >= 10):
   date = str(t1.year) + "-0" + str(t1.month) + "-" + str(t1.day)
elif (t1.month >= 10 and t1.day < 10):
   date = str(t1.year) + "-" + str(t1.month) + "-0" + str(t1.day)
else:
   date = str(t1.year) + "-" + str(t1.month) + "-" + str(t1.day)

time = str(t1.hour) + ":" + str(t1.minute) + ":" + str(t1.second)
records = []
i = 0
for i in xrange(0, ROW):
    #if (int(Data[i,1]) > 1.25 * (int(Data[i,12]))) and ((int(Data[i,12])) != 0):
    if(int(Data[i,23]) == 1):
        records.append([str(Data[i,0]),date,time,int(Data[i,1]),int(Data[i,2]),int(Data[i,3]),int(Data[i,4]),float(Data[i,5]),float(Data[i,6]),float(Data[i,7]),int(Data[i,8]),int(Data[i,9]),float(Data[i,10]),float(Data[i,11]),int(Data[i,12]),int(Data[i,13]),int(Data[i,14]),int(Data[i,15]),float(Data[i,16]),float(Data[i,17]),float(Data[i,18]),int(Data[i,19]),int(Data[i,20]),float(Data[i,21]),float(Data[i,22])])
input_table.insert_records(records)

