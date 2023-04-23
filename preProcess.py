import pandas as pd, numpy as np
import random

def status_decorator(func):
    def decorate(*params,status=""):
        print(status,end="")
        func(*params)
        print("Done")
    return decorate

def virus_location(virus_name):
    if virus_name.startswith("montreal"): return "montreal"
    if virus_name.startswith("padua"): return "padua"
    if virus_name.startswith("princeton"): return "princeton"
    return virus_name

def ransom(virus_name):
    if virus_name=="white": return virus_name
    else: return "ransom"

codes = {0:"white",1:"montreal",2:"padua",3:"princeton"}
backcodes = {"white":0,"montreal":1,"padua":2,"princeton":3}

#codes = {0:"white",1:"ransom"}
#backcodes = {"ransom":1,"white":0}

class PreProcessor:

    def __init__(self,dataframe,dim=None):
        self.df = dataframe
        self.df.rename(columns={"count":"counts"},inplace=True)

    def preprocessing(self):
        self.random_undersampling()
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
            self.df.fillna(self.df.median(),inplace=True)
            self.df.dropna(inplace=True)

    @status_decorator
    def dim_reduction(self,dim=None):
        print("Reducing dimensions to ",end="")
        self.meanframe = self.df.mean()
        self.stdframe = self.df.std(ddof=0,numeric_only=True)
        features = self.meanframe.index
        #normalising the values and adding them at the end
        for feature in features:
            self.df['N'+feature] = (self.df[feature] - self.meanframe[feature])/self.stdframe[feature]

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
                if proportion_of_variance > 0.80: break
                dim += 1

        print("%d..."%dim,end="")
        self.V = eig_vecs[:,:dim]

        #Calculate the projection of original/normalised("N") vectors in new directions
        new_space = (self.df["N"+features]@self.V).add_prefix("F")

        self.df.drop(features,axis=1,inplace=True)
        self.df.drop("N"+features,axis=1,inplace=True)

        self.df = pd.concat([self.df,new_space],axis=1)

    def random_undersampling(obj,thresh=1.25):
        # Threshold = 25% of mean of samples. If the majority class exceeds this, it'll be sampled
        print("Undersampling the data...",end="")
        while True:
            class_table = obj.df.label.value_counts()
            mean_samples_per_class = class_table.mean()
            
            if class_table[0]<=round(1.25*mean_samples_per_class): break

            majority = (obj.df.label==class_table.index[0])
            majority_index = majority[majority].index

            sampling_index = random.sample(
                                list(majority_index),
                                int(round(class_table[0] - thresh*mean_samples_per_class)),
                                )
            obj.df.drop(sampling_index,inplace=True)

        obj.df.reset_index(drop=True,inplace=True)
        data = len(obj.df)
        print("finished undersampling.\nReduced data:",data)
        return data
