import sys
import collections
from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbColumnProperty, GPUdbTableOptions
import numpy as np
import datetime
import time
from time import sleep

loop = 500
KINETICA_HOST = '10.20.10.228'
KINETICA_PORT = '9191'
INPUT_TABLE = "SUMMARY_STATISTICS"


h_db = GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

COLUMNS =["Toplam_Konusma_Suresi_dk","Toplam_Arama_Sayisi","Ortalama_Konusma_Suresi_dk", "Baska_Ulkeyi_Arama_Orani","Gun_ici_Arama_Orani","Aksam_Arama_Orani","Gece_Arama_Orani","Toplam_Aranma_Sayisi","Ortalama_Konusma_Suresi_aranma_dk","Cevaplama_Orani","Baska_Ulkeden_Arama_Orani","pd_Toplam_Konusma_Suresi_dk","pd_Toplam_Arama_Sayisi","pd_Ortalama_Konusma_Suresi_dk", "pd_Baska_Ulkeyi_Arama_Orani","pd_Gun_ici_Arama_Orani","pd_Aksam_Arama_Orani","pd_Gece_Arama_Orani","pd_Toplam_Aranma_Sayisi","pd_Ortalama_Konusma_Suresi_aranma_dk","pd_Cevaplama_Orani","pd_Baska_Ulkeden_Arama_Orani","Is_Fraud"]
ROW = 50
Width = len(COLUMNS)
#InputTrain = h_db.get_records_by_column(table_name = INPUT, column_names = COLUMNS, offset = 0,  limit = ROW,encoding = "json")
#Train_Input = np.array(h_db.parse_dynamic_response(InputTrain)['response'].values(),dtype='f')
#Train_Input = np.transpose(Train_Input)

columns = []
columns.append(GPUdbRecordColumn("Tablo_Ad", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Variable", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Time", GPUdbRecordColumn._ColumnType.STRING))
columns.append(GPUdbRecordColumn("Date", GPUdbRecordColumn._ColumnType.STRING,[GPUdbColumnProperty.DATE]))
columns.append(GPUdbRecordColumn("Min", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Max", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Range", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("1st_Q", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("2nd_Q", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("3rd_Q", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Median", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Average", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Mean", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Std_dev", GPUdbRecordColumn._ColumnType.FLOAT))
columns.append(GPUdbRecordColumn("Variance", GPUdbRecordColumn._ColumnType.FLOAT))

#if h_db.has_table(table_name = INPUT_TABLE)['table_exists']:
#   h_db.clear_table(table_name = INPUT_TABLE)
input_table = GPUdbTable(columns, INPUT_TABLE, db = h_db)
INPUT_TAB = "CDR_DATA_HIST"

for j in range(1,loop):
    for i in range(1,4):
        INPUT = INPUT_TAB + str(i)
	InputTrain = h_db.get_records_by_column(table_name = INPUT, column_names = COLUMNS, offset = 0,  limit = ROW,encoding = "json")
	Train_Input = np.array(h_db.parse_dynamic_response(InputTrain)['response'].values(),dtype='f')
	Train_Input = np.transpose(Train_Input)

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
        for i in range(0, Width):
            records.append([INPUT, COLUMNS[i], time, date, float(np.amin(Train_Input[:, i])), float(np.amax(Train_Input[:, i])),
                            float(np.ptp(Train_Input[:, i])), float(np.percentile(Train_Input[:, i], 25)),
                            float(np.percentile(Train_Input[:, i], 50)), float(np.percentile(Train_Input[:, i], 75)),
                            float(np.median(Train_Input[:, i])), float(np.average(Train_Input[:, i])),
                            float(np.mean(Train_Input[:, i])), float(np.std(Train_Input[:, i])),
                            float(np.var(Train_Input[:, i]))])
    
        input_table.insert_records(records)
    sleep(60)
