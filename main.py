#!/usr/bin/env python3

import numpy as np
import pandas as pd

from preProcess import *

DATASET = "try.csv"
#DATASET = "BitcoinHeistData.csv"

def split_dataset(data,perc=.80):
    # 80 - 20 % is Pareto principle by default
    global tr,te,vl
    print("Spliting data...",end="")

    n = len(data)
    train_len = round(n*perc*perc)
    test_len = round(n-n*perc)

    tr = data[:train_len]
    te = data[train_len:train_len+test_len]
    vl = data[train_len+test_len:]

    te.reset_index(drop=True,inplace=True)
    vl.reset_index(drop=True,inplace=True)
    print("Done")
    
def main():
    df = pd.read_csv(DATASET)
    split_dataset(df)
    del df
    preprocess(tr)
    tr.info()
    print(tr)

if __name__=="__main__":
    main()
