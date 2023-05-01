import pandas as pd, numpy as np

from preProcess import *

class Predictor:

    def __init__(self,preprocessor,classifier):
        self.cfy = classifier
        self.pp = preprocessor

    def take_test(self,test_df):
        #perform renaming and changing datatypes
        self.test_data = PreProcessor(test_df)
        self.test_data.change_dtypes()
        self.features = test_df.columns[(test_df.dtypes==np.float64) | (test_df.dtypes==np.int64)]

        self.dim_red() #self.red DF is created with reduced features
        self.new_features = self.red.columns[(self.red.dtypes==np.float64) | (self.red.dtypes==np.int64)]


    def mlr(self,testdf=None):
        """
        Using multiple linear regression
        """
        if testdf: self.take_test(testdf)

        X = np.c_[self.red[self.new_features],np.ones((len(self.red),1),np.int64)]
        Y = X @ self.cfy.mr_w

        def wrap(num):
            if 0<= num <=3: return num
            return 0
        #Y = (Y - Y.mean())/Y.std()
        self.red['predicted'] = Y
        self.red.predicted = self.red.predicted.apply(round).apply(wrap)

        self.create_confusion_matrix()


    def ols(self,testdf=None):
        if testdf: self.take_test(testdf)
        
        X = self.red[self.new_features]
        Y = X @ self.cfy.ol_w

        def wrap(num):
            if 0<= num <=3: return num
            return 0

        self.red['predicted'] = Y
        self.red.predicted = self.red.predicted.apply(round).apply(wrap)

        self.create_confusion_matrix()

    def knn(self,testdf=None,k=3):
        if testdf: self.take_test(testdf)

        self.cfy.knn(self.red[self.new_features].iloc[0].to_numpy())
        prediction = []
        for i in range(len(self.red)):
            label = self.cfy.knn(self.red[self.new_features].iloc[i].to_numpy())
            self.red['predicted'] = backcodes[label]

        self.create_confusion_matrix()

    def dim_red(self):
        """
        This function is used to reduce dimension of the test set, using the
        same transformation matrix used for training set
        """
        self.red = None
        for feature in self.features:
            self.test_data.df['S'+feature] = (self.test_data.df[feature] - self.pp.meanframe[feature])/self.pp.stdframe[feature]

        new_space = (self.test_data.df["S"+self.features] @ self.pp.V).add_prefix("F")
        new_space['original'] = self.test_data.df.original
        self.red = new_space

    def create_confusion_matrix(self):
        # Dict are unordered, so creating an ordered tuple here
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

    def get_score(self):
        """
        We're interested in ransom payments, so white is negative class, and other than white
        are positive class
        """
        cm = self.cm.to_numpy(dtype=np.int64)
        total = sum(sum(cm))


        #Print a formated result
        print("="*50)
        for i in codes:
            tp = cm[i][i]
            row_sum = sum(cm[i,:])
            col_sum = sum(cm[:,i])
            fp = col_sum - tp
            fn = row_sum - tp
            tn = total - (tp + fp + fn)

            #Calculate the values
            accuracy    = (tp+tn)/(tp+tn+fp+fn)
            recall      = (tp)/(tp+fn)
            precision   = (tp)/(tp+fp)
            if any((recall,precision)):
                F1_score = 2*recall*precision/(recall+precision)
            else: F1_score = 0.0

            #Print the result
            print("Score for",codes[i])
            print("\tAccuracy:\t",accuracy)
            print("\tPrecision:\t",precision)
            print("\tRecall:\t\t",recall)
            print("\tF1-Score:\t",F1_score)

            print("="*50)

