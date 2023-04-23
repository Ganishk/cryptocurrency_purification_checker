import pandas as pd, numpy as np


class Classifier:

    def __init__(self,dataframe):
        self.df = dataframe
        self.categories = self.df.label.cat.categories
        self.features = self.df.columns[(self.df.dtypes==np.float64) | (self.df.dtypes==np.int64)]

        self.classify()

    def classify(self):
        pass
