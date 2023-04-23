import pandas as pd, numpy as np

from preProcess import *

class Predictor:

    def __init__(self,preprocessor,classifier):
        self.cfy = classifier
        self.pp = preprocessor

    def predict(self,test_df):
        #perform renaming and changing datatypes
        self.test_data = PreProcessor(test_df)
        self.test_data.change_dtypes()
        self.features = test_df.columns[(test_df.dtypes==np.float64) | (test_df.dtypes==np.int64)]

        self.dim_red()
        self.new_features = self.red.columns[(self.red.dtypes==np.float64) | (self.red.dtypes==np.int64)]


        X = np.c_[self.red[self.new_features],np.ones((len(self.red),1),np.int64)]
        Y = X @ self.cfy.w

        def wrap(num):
            if 0<= num <=3: return num
            return 0
        #Y = (Y - Y.mean())/Y.std()
        self.red['predicted'] = Y
        self.red.predicted = self.red.predicted.apply(round).apply(wrap)

        self.create_confusion_matrix()


    def dim_red(self):
        for feature in self.features:
            self.test_data.df['N'+feature] = (self.test_data.df[feature] - self.pp.meanframe[feature])/self.pp.stdframe[feature]

        new_space = (self.test_data.df["N"+self.features] @ self.pp.V).add_prefix("F")
        new_space['original'] = self.test_data.df.label.apply(lambda x: backcodes[x])
        self.red = new_space

    def create_confusion_matrix(self):
        classes = ("white","montreal","padua","princeton")
        mClass = tuple(map(lambda x: backcodes[x],classes))
        self.cm = np.eye(len(classes))
        for true_class in mClass:
            for predicted_class in mClass:
                self.cm[true_class][predicted_class] = len(self.red[
                                (self.red.predicted == predicted_class) &
                                (self.red.original == true_class) ])

        classes = pd.Series(classes)
        self.cm = pd.DataFrame(self.cm,index="True_"+classes,columns="P_"+classes,dtype=np.int64)
