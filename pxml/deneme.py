from __future__ import print_function

import collections
import json
import random
import string
import numpy as np
import gpudb
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree import ElementTree
from xml.dom import minidom
import time
import multiprocessing as mp

def retrieve():
    global i
    KINETICA_HOST = '10.20.10.228'
    KINETICA_PORT = '9191'
   
    h_db = gpudb.GPUdb(encoding = 'BINARY', host = KINETICA_HOST, port = KINETICA_PORT)

    response = h_db.get_records(table_name = "udf_json_example",offset = 0,  limit = 1,encoding = "binary") 
    tablo = gpudb.GPUdbRecord.decode_binary_data(response["type_schema"],response["records_binary"])
    #response = h_db.get_records(table_name = "udf_json_example",offset = 0,  limit = 2,encoding = "json")['records_json']
    print(response)
    #tablo = gpudb.GPUdbRecord.decode_json_string_data(response)

    print(tablo)
    main = Element('XML ORNEK')
    i = 1
    for xml in tablo:
        child0 = SubElement(main,'Index')
        child0.text = i.__str__()

        child1 = SubElement(main,'Domain')
        child1.text = "{domain}".format(**xml) 
            
        child2 = SubElement(main,'Code')
        child2.text = "{code}".format(**xml)
        i += 1
    sadi = ElementTree.ElementTree(main)
    sadi.write("output.xml")
if __name__ == '__main__':
   a = time.time()
   retrieve()
   b = time.time()
   print(b-a,i) 
