import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def read_uci(dataset,stats=False):
    path = f'/home/shihab/Datasets/UCI/{dataset}.txt'
    df = pd.read_csv(path,delim_whitespace=True,header=None)
    df = df.astype('float64')
    data = df.values
    X,Y = data[:,1:],data[:,0].astype('int32')
    if Y.min()==1:
        Y -= 1
    X = MinMaxScaler().fit_transform(X)
    if stats:
        labels,freq = np.unique(Y,return_counts=True)
        print(dataset,X.shape,len(labels),freq.min()/freq.max(),freq)
    return shuffle(X,Y,random_state=42)

def info():
	for f in sorted(os.listdir("/home/shihab/Datasets/UCI/")):
	    if not f.endswith('txt'): continue
	    try:
	        X,Y = read_uci(f.split('.')[0],True)
	    except Exception as e:
	        print("ERROR:",f,e)

info()
