import pandas as pd, numpy as np
import random

def status_decorator(func):
    """
    This function is used to decorate the other functions
    by showing whether it has ended or not
    """
    def decorate(*params,status=""):
        print(status,end="")
        func(*params)
        print("Done")
    return decorate

def virus_location(virus_name):
    """
    This function is used to return the location of virus.
    Usually it is used to change the class labels, when we're
    intesrested in finding where the virus came from.
    """
    if virus_name.startswith("montreal"): return "montreal"
    if virus_name.startswith("padua"): return "padua"
    if virus_name.startswith("princeton"): return "princeton"
    return virus_name

def ransom(virus_name):
    """
    This function is used to return whether given payment
    is an ransom or not.
    """
    if virus_name=="white": return virus_name
    else: return "ransom"

# The following codes are used to encode and decode the
# class labels when needed.
codes = {0:"white",1:"montreal",2:"padua",3:"princeton"}
backcodes = {"white":0,"montreal":1,"padua":2,"princeton":3}

# The following codes is used for binary classification,
# encoding and decoding of data.
#codes = {0:"white",1:"ransom"}
#backcodes = {"ransom":1,"white":0}

class PreProcessor:
    """
    Class to contain the data needed for preprocessing
    """
    def __init__(self,dataframe,dim=None):
        """
        Contructor of the PreProcessor Class
        @params
            dataframe:
                DataFrame that is needed to preprocess it's data.
            dim:
                No. of dimensions needed to retain after dimensionality
                reduction.
        """
        self.df = dataframe
        self.df.rename(columns={"count":"counts"},inplace=True)
        self.dim = dim

    def preprocessing(self):
        """
        Steps in preprocessing
        """
        self.random_undersampling()
        self.clean_data()
        self.change_dtypes()
        self.dim_reduction(self.dim)

    @status_decorator
    def change_dtypes(self):
        """
        This function is used to change the values to desired
        datatype.
        """
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
        self.df['original'] = self.df.label.apply(lambda x: backcodes[x])

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
        """
        Dimensionality reduction by Principal component Analysis
        """
        print("Reducing dimensions to ",end="")

        # For standardising values
        self.meanframe = self.df.mean()
        self.stdframe = self.df.std(ddof=0,numeric_only=True)

        # For normalising values
        self.minframe = self.df.min()
        self.maxframe = self.df.max()

        features = self.meanframe.index
        # the values and adding them at the end
        for feature in features:
            self.df['S'+feature] = (self.df[feature] - self.meanframe[feature])/self.stdframe[feature]
            #self.df["N"+feature] = (self.df[feature] - self.minframe[feature])/(
            #        self.maxframe[feature] - self.minframe[feature])

        covariance_matrix = self.df["S"+features].cov()
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

        #Calculate the projection of original/standardised("S") vectors in new directions
        new_space = (self.df["S"+features]@self.V).add_prefix("F")

        self.df.drop(features,axis=1,inplace=True)
        self.df.drop("S"+features,axis=1,inplace=True)

        self.df = pd.concat([self.df,new_space],axis=1)

    def random_undersampling(obj,thresh=1.25):
        """
        This function is used to random undersampling to unbias the data
        """
        # Threshold = 25% of mean of samples. If the majority class exceeds this, it'll be sampled
        print("Undersampling the data...",end="")
        datai=len(obj.df)
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
        print("finished undersampling.\nReduced data from",datai,"to",data)
        return data
