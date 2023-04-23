#!/usr/bin/env python3

import argparse
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from preProcess import *
from model import *

#DATASET = "try.csv"

@status_decorator
def split_dataset(data,perc=.80):
    # 80 - 20 % is Pareto principle by default
    global tr,te,vl
    print("Spliting data...",end="")

    # Below line can be removed or commented to be more specific about the virus
    data.label = data.label.apply(virus_location)

    n = len(data)
    train_len = round(n*perc*perc)
    test_len = round(n-n*perc)

    tr = data[:train_len]
    te = data[train_len:train_len+test_len]
    vl = data[train_len+test_len:]

    te.reset_index(drop=True,inplace=True)
    vl.reset_index(drop=True,inplace=True)
    
def main(args=None):
    DATASET = args.dataset

    df = pd.read_csv(DATASET)
    split_dataset(df)
    del df

    # Preprocessing Data
    if args.preprocess:
        preprocess = pickle.load(args.preprocess)
        args.preprocess.close()
    else:
        preprocess = PreProcessor(tr)
        with open(DATASET[:-4]+'preprocessed.obj',"wb") as file: pickle.dump(preprocess,file)

    class_table = preprocess.df.label.value_counts()
    class_table.plot(kind='bar',title="Distribution of class table")
    print("\nCurrent distribution of training data:")
    print(class_table)


    classifier = Classifier(preprocess.df)
    print("\nCategories:")
    print(classifier.categories)

    features = te.columns[(te.dtypes==np.float64) | (te.dtypes==np.int64)]


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CSOE18 ML Project")

    parser.add_argument('-d','--dataset',type=str,required=True)
    parser.add_argument('-p','--preprocess',type=argparse.FileType('rb'))
    parser.add_argument('-m','--model',type=argparse.FileType('rb'))
    parser.add_argument('-r','--result',metavar="nooooo",type=argparse.FileType('rb'))

    main(parser.parse_args())
    plt.show()
