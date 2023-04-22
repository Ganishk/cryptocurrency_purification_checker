import pandas as pd, numpy as np


class Classifier:

    def __init__(self,dataframe):
        self.df = dataframe
        self.categories = self.df.label.cat.categories
        self.features = self.df.columns[(self.df.dtypes==np.float64) | (self.df.dtypes==np.int64)]

        self.classify()

    def classify(self):
        X = self.df[self.features]
        self.codes = dict(zip(self.df.label.cat.codes,self.df.label))
        self.w = np.linalg.inv( X.transpose() @ X ) @ X.transpose() @ self.df.label.cat.codes
