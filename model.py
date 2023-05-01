import pandas as pd, numpy as np

from preProcess import codes,backcodes

class Classifier:

    def __init__(self,dataframe):
        self.df = dataframe
        self.categories = self.df.label.cat.categories
        self.features = self.df.columns[(self.df.dtypes==np.float64) | (self.df.dtypes==np.int64)]

        self.ols()
        self.multiple_regression()
        self.naive_bayes()


    def multiple_regression(self):
        #multiple linear regression
        X = np.c_[self.df[self.features],np.ones((len(self.df),1),np.int64)]
        self.mr_w = np.linalg.inv( X.transpose() @ X ) @ X.transpose() @ self.df.label.apply(lambda x: backcodes[x])

    def ols(self):
        #Ordinary Least Square method
        X = self.df[self.features]
        self.ol_w = (np.linalg.inv( X.transpose() @ X ) @ X.transpose() @ self.df.label.apply(lambda x: backcodes[x])).to_numpy()

    def knn(self,X,k=3):
        """
        Here, we're using Euclidean distance || X0 - X1 ||
        But for large dataset knn is time consuming
        It's preferred that k should be odd to avoid tie
        """
        distance = lambda x: np.linalg.norm(X - x)
        self.df['distance'] = self.df[self.features].apply(distance,axis=1)
        idx = self.df.distance.argsort()[:3]
        result = self.df.label.iloc[idx]
        return result.mode()[0]

    def naive_bayes(self):
        # Gaussian Naive Bayes
        self.priors = {} # P(Ci) - Prior probabilities
        self.posteriors = {} # P(Ci|X)
        n = self.df.shape[0]
        for virus in backcodes:
            f_virus = self.df[self.df.label == virus]
            self.priors[virus] = f_virus.shape[0]/n
            self.posteriors[virus] = {}
            for feature in self.features:
                self.posteriors[virus][feature] = {
                        'mean': f_virus[feature].mean(),
                        'std': f_virus[feature].std()
                    }
