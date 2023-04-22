import pandas as pd, numpy as np

def status_decorator(func):
    def decorate(*params,status=""):
        print(status,end="")
        func(*params)
        print("Done")
    return decorate

class PreProcessor:

    def __init__(self,dataframe,dim=None):
        self.df = dataframe
        self.df.rename(columns={"count":"counts"},inplace=True)

        self.clean_data()
        self.change_dtypes()
        self.dim_reduction()

    @status_decorator
    def change_dtypes(self):
        print("Changing datatypes...",end="")
        self.df.address   = self.df.address.astype('object')
        self.df.year      = self.df.year.astype(np.int64)
        self.df.day       = self.df.day.astype(np.int64)
        self.df.length    = self.df.length.astype(np.int64)
        self.df.weight    = self.df.weight.astype(np.float64)
        self.df.counts    = self.df.counts.astype(np.int64)
        self.df.looped    = self.df.looped.astype(np.int64)
        self.df.neighbors = self.df.neighbors.astype(np.int64)
        self.df.income    = self.df.income.astype(np.int64)
        self.df.label     = pd.Categorical(self.df.label)

    @status_decorator
    def clean_data(self):
        """
        Here, we're using median as a central tendency. In a highly skewed dataset
        like this, there is a little difference between the median and mean.
        Also, it preserves the datatype of that column if it is a type of integer,
        for which using mean might introduce a float value
        """
        print("Cleaning data...",end="")
        if self.df.isnull().values.any():
            self.df.fillna(dataframe.median(),inplace=True)
            self.df.dropna(inplace=True)

    @status_decorator
    def dim_reduction(self,dim=None):
        print("Reducing dimensions to ",end="")
        meanframe = self.df.mean()
        stdframe = self.df.std(ddof=0,numeric_only=True)
        features = meanframe.index
        #normalising the values and adding them at the end
        for feature in features:
            self.df['N'+feature] = (self.df[feature] - meanframe[feature])/stdframe[feature]

        covariance_matrix = self.df["N"+features].cov()
        eig_vals,eig_vecs = np.linalg.eig(covariance_matrix)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]

        if not dim:
            dim = 0
            # Get the dimension required for 90% information
            E = sum(eig_vals)
            while True:
                proportion_of_variance = sum(eig_vals[:dim])/E
                if proportion_of_variance > 0.90: break
                dim += 1

        print("%d..."%dim,end="")
        self.V = eig_vecs[:,:dim]

        #Calculate the projection of original vectors in new directions
        new_space = (self.df[features]@self.V).add_prefix("F")

        self.df.drop(features,axis=1,inplace=True)
        self.df.drop("N"+features,axis=1,inplace=True)

        self.df = pd.concat([self.df,new_space],axis=1)

