import KineticaIO


PROC_NAME = 'UDF_Mnist_train'
DISTRIBUTED_MODE = 'nondistributed'
UDF_MAIN = 'TrainUDF.py'
FILE_LIST = [UDF_MAIN, 'KineticaIO.py', 'DatabaseDefinitions.py']


if __name__ == "__main__":
    kio = KineticaIO.KineticaIO()
    kio.register_udf(proc_name=PROC_NAME, distributed_mode=DISTRIBUTED_MODE, file_list=FILE_LIST, sysargs=[UDF_MAIN])
    kio.execute_udf(proc_name=PROC_NAME)
