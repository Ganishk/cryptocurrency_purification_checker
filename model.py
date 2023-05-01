import pandas as pd, numpy as np

from preProcess import codes,backcodes

class Classifier:

    def __init__(self,dataframe):
        self.df = dataframe
        self.categories = self.df.label.cat.categories
        self.features = self.df.columns[(self.df.dtypes==np.float64) | (self.df.dtypes==np.int64)]

        self.multiple_regression()
        self.ols()


    def multiple_regression(self):
        #multiple linear regression
        g = np.c_[self.df[self.features],np.ones((len(self.df),1),np.int64)]
        self.mr_w = np.linalg.inv( g.transpose() @ g ) @ g.transpose() @ self.df.label.apply(lambda x: backcodes[x])

    def ols(self):
        #Ordinary Least Square method
        g = self.df[self.features]
        self.ol_w = np.linalg.inv( g.transpose() @ g ) @ g.transpose() @ self.df.label.apply(lambda x: backcodes[x])
