#!/usr/bin/env python3

import argparse
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from preProcess import *
from model import *
from predict import *

@status_decorator
def split_dataset(data,perc=.80):
    # 80 - 20 % is Pareto principle by default
    global tr,te,vl,codes
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

def scatter_graph():
    global preprocess,classification
    a = "F0","F1","F2","F3","F4","F5"
    for x in range(len(a)):
        plt.rcParams["figure.figsize"] = 12,8
        plt.figure()
        for y in range(x+1,len(a)):
            plt.subplot(2,3,y)
            plt.scatter(preprocess.df[a[x]][classification=="montreal"],preprocess.df[a[y]][classification=="montreal"],label="montreal")
            plt.scatter(preprocess.df[a[x]][classification=="padua"],preprocess.df[a[y]][classification=="padua"],label="padua")
            plt.scatter(preprocess.df[a[x]][classification=="princeton"],preprocess.df[a[y]][classification=="princeton"],label="princeton")
            plt.scatter(preprocess.df[a[x]][classification=="white"],preprocess.df[a[y]][classification=="white"],label="white")
            plt.legend()

   
def main(args=None):
    DATASET = args.dataset

    df = pd.read_csv(DATASET)
    split_dataset(df)
    del df

    # Preprocessing Data
    if args.preprocess:
        preprocess = pickle.load(args.preprocess)
        args.preprocess.close()
    #elif args.model or args.result: pass
    else:
        preprocess = PreProcessor(tr)
        preprocess.preprocessing()

        with open(DATASET[:-4]+'_preprocessed.obj',"wb") as file: pickle.dump(preprocess,file)
        print("\n[+] Preprocessed data was saved")

        classification = preprocess.df.label
        class_table = classification.value_counts()
        class_table.plot(kind='bar',title="Distribution of class table")

    # Training On Data
    if args.model:
        classifier = pickle.load(args.model)
        args.model.close()
    else:
        classifier = Classifier(preprocess.df)
        with open(DATASET[:-4]+'_model.obj','wb') as file: pickle.dump(classifier,file)
        print("\n[+] Classifier data was saved")

        #print("\nCategories:")
        #print(classifier.categories)

    features = te.columns[(te.dtypes==np.float64) | (te.dtypes==np.int64)]
    predictor = Predictor(preprocess,classifier)

    predictor.predict(vl)

    print("\nConfusion Matrix")
    print(predictor.cm)

    #scatter_graph()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CSOE18 ML Project")

    parser.add_argument('-d','--dataset',type=str,required=True)
    parser.add_argument('-p','--preprocess',type=argparse.FileType('rb'))
    parser.add_argument('-m','--model',type=argparse.FileType('rb'))
    parser.add_argument('-r','--result',metavar="nooooo",type=argparse.FileType('rb'))

    main(parser.parse_args())
    plt.show()
