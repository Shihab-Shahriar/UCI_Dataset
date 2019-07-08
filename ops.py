import os
import pandas as pd
import numpy as np

def read_uci(dataset):
    """No Preprocessing applied"""
    path = f'/home/shihab/Datasets/UCI/{dataset}.txt'
    df = pd.read_csv(path,delim_whitespace=True,header=None)
    df = df.astype('float64')
    data = df.values
    X,Y = data[:,1:],data[:,0]
    return X,Y

def info():
	for f in sorted(os.listdir("/home/shihab/Datasets/UCI/")):
	    if not f.endswith('txt'): continue
	    try:
	        X,Y = read_uci(f.split('.')[0])
	        c,cs = np.unique(Y,return_counts=True)
	        print(f"{f},Shape: {X.shape}, classes: {len(c)}, freq: {cs}")
	    except Exception as e:
	        print("ERROR:",f,e)

info()
