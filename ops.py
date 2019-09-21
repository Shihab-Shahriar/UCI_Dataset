import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

def read_uci(dataset,stats=False):
    path = f'~/Datasets/UCI/{dataset}.txt'
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

def accuracy():
    df = pd.DataFrame()
    cv = RepeatedStratifiedKFold(n_repeats=5,n_splits=5,random_state=42)
    for f in sorted(os.listdir()):
        if not f.endswith('txt'): continue
        name = f.split('.')[0]
        X,Y = read_uci(name,True)
        r = cross_val_score(RandomForestClassifier(n_estimators=1000,n_jobs=-1),X,Y,cv=cv)*100
        df.at[name,'result'] = f"{r.mean():.2f}_{r.std():.2f}"
        print(name, df.loc[name,'result'])
    return df

df = accuracy()
df.to_csv("accuracy.csv")