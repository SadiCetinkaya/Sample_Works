import sys,os,os.path
from kinetica_proc import ProcData
from skcuda import cublas
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

# Update PATH with location of nvcc compiler
CUDA_BIN = '/usr/local/cuda/bin'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.sep,os.sep.join(CUDA_BIN.split('/')))
#Set XDG_CACHE_HOME directory for PyCUDA to use for temp files;
#will use HOME directory for temp space if not set, which for gpudb_proc
#is /home/gpudb, where it doesn't have write access

os.environ['XDG_CACHE_HOME'] = '/tmp'

def cublas_add_vectors(h,x,y):
   x_gpu = gpuarray.to_gpu(x)
   y_gpu = gpuarray.to_gpu(y)
   cublas.cublasSaxpy(h, x.size, 1.0, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
   return y_gpu.get()

def example(pd):
   for in_table, out_table in zip(pd.input_data, pd.output_data):      
      out_table.size = in_table.size
      z = out_table[0]
      x = np.ndarray(shape=(in_table.size, 1), dtype=float).astype(np.float32)
      y = np.ndarray(shape=(in_table.size, 1), dtype=float).astype(np.float32)
      w = np.ndarray(shape=(1, in_table.size), dtype=float).astype(np.float32)
      # Initialize vectors & matrix with database values
      for i in xrange(0, in_table.size):
         x[i,0] = in_table['x'][i]
         y[i,0] = in_table['y'][i]

      h = cublas.cublasCreate()
      w = cublas_add_vectors(h,x,y)
      z = w
      cublas.cublasDestroy(h)
   
if __name__ == "__main__":

   proc_data = ProcData()

   if int(proc_data.request_info["data_segment_number"]) + 1 == int(proc_data.request_info["data_segment_count"]):
       example(proc_data)

   proc_data.complete()

