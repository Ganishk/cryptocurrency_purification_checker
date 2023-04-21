#!/usr/bin/env python3

import numpy as np
import pandas as pd

from preProcess import *

DATASET="try.csv"

def split_dataset(data,perc=.80):
    # 80 - 20 % is Pareto principle by default
    global tr,te,vl
    print("Spliting data...")

    n = len(data)
    train_len = round(n*perc*perc)
    test_len = round(n-n*perc)

    tr = data[:train_len]
    te = data[train_len:train_len+test_len]
    vl = data[train_len+test_len:]
    
def main():
    df = pd.read_csv(DATASET)
    split_dataset(df)
    del df
    preprocess(tr)
    print(tr.info())
    print('\nmean\n%s'%tr.mean())
    print('\nvariance\n%s'%tr.var())
    print('\nskewness\n%s'%tr.skew())
    print('\nkurtosis\n%s'%tr.kurtosis())

if __name__=="__main__":
    main()
