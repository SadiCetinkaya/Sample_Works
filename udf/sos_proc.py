################################################################################
#                                                                              #
# Kinetica UDF Sum of Squares Example UDF                                      #
# ---------------------------------------------------------------------------- #
# This UDF takes pairs of input & output tables, computing the sum of the      #
# squares of all the columns for each input table and saving the resulting     #
# sums to the first column of the corresponding output table.                  #
#                                                                              #
################################################################################
import sys
import math
from itertools import islice
from kinetica_proc import ProcData

proc_data = ProcData()

# For each pair of input & output tables, calculate the sum of squares of input
#    columns and save results0 to first output table column
for in_table, out_table in zip(proc_data.input_data, proc_data.output_data):

   # Extend the output table by the number of record entries in the input table
   out_table.size = in_table.size

   # Use the second column in the output table as the sum column
   y = out_table[1]

   # For every record in the table...
   for i in xrange(0, in_table.size):
      # Copy the input ID to the output ID for later association
      out_table[0][i] = in_table[0][i]

   # Loop through the remaining input table columns
   for in_column in islice(in_table, 1, None):
      # For every value in the column...
      for calc_num in xrange(0, in_table.size):
         # Add the square of that value to the corresponding output column
         y[calc_num] += in_column[calc_num] ** 2

proc_data.complete()

