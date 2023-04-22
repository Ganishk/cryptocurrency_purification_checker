#!/usr/bin/env python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from preProcess import *

DATASET = "try.csv"
#DATASET = "BitcoinHeistData.csv"

@status_decorator
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
    
def main():
    df = pd.read_csv(DATASET)
    split_dataset(df)
    del df
    preprocess = PreProcessor(tr)

    class_table = preprocess.df.label.value_counts()
    class_table.plot(kind='bar',title="Distribution of class table")
    print("Current distribution of training data")
    print(class_table)

if __name__=="__main__":
    main()
    plt.show()
