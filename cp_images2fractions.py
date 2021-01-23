import os
import shutil 
import pandas as pd
import numpy as np

data_list_path = './data_list'
csvfiles = os.listdir(data_list_path)
for i,fn in enumerate(csvfiles):
    if fn.startswith('fraction') and '0.50' in fn:
        fn_abs = os.path.join(data_list_path, fn)
        new_fn = fn_abs.replace('0.50', '15')
        print(i, fn_abs)
        os.rename(fn_abs, new_fn)
        print(i, new_fn)
     
