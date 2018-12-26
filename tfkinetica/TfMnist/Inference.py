import KineticaIO
import DatabaseDefinitions as db


GPU_MEMORY_ALLOCATION_FRACTION = "0.1"  # if Kinetica is running on CPU instance, this parameter has no effect

PROC_NAME = 'UDF_Mnist_inference'
DISTRIBUTED_MODE = 'distributed'
UDF_MAIN = 'InferenceUDF.py'
FILE_LIST = [UDF_MAIN, 'KineticaIO.py', 'DatabaseDefinitions.py']  # main python script has to be in first position


if __name__ == "__main__":
    kio = KineticaIO.KineticaIO()
    kio.register_udf(proc_name=PROC_NAME, distributed_mode=DISTRIBUTED_MODE, file_list=FILE_LIST,
                     sysargs=[UDF_MAIN, GPU_MEMORY_ALLOCATION_FRACTION])
    kio.execute_udf(proc_name=PROC_NAME, input_table_names=[db.INFERENCE_INPUT_TABLE],
                    output_table_names=[db.INFERENCE_OUTPUT_TABLE])


# confusion matrix:
# select Predict, Label, count(*) as Count from CreditDefault.Credit_inference_result group by Predict, Label
# multiple models:
# SELECT Id, sum(Predict) AS DefaultVote, min(Label) AS Label
# FROM CreditDefault.Credit_inference_result GROUP BY Id ORDER BY Id ASC limit 50
#
# SELECT DefaultVote, Label, count(Label) AS CntLabel FROM
# (SELECT Id, sum(Predict) AS DefaultVote, min(Label) AS Label
#   FROM CreditDefault.Credit_inference_result GROUP BY Id ORDER BY Id ASC LIMIT 50000)
# GROUP BY DefaultVote, Label ORDER BY Label, DefaultVote ASC
