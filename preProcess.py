import pandas as pd, numpy as np

def status_decorator(func):
    def decorate(*params,status=""):
        print(status,end="")
        func(*params)
        print("Done")
    return decorate

@status_decorator
def change_dtypes(df):
    print("Changing datatypes...",end="")
    df.address  = df.address.astype('object')
    df.year     = df.year.astype(np.int64)
    df.day      = df.day.astype(np.int64)
    df.length   = df.length.astype(np.int64)
    df.weight   = df.weight.astype(np.float64)
    df.counts   = df.counts.astype(np.int64)
    df.looped   = df.looped.astype(np.int64)
    df.neighbors= df.neighbors.astype(np.int64)
    df.income   = df.income.astype(np.int64)
    df.label    = pd.Categorical(df.label)

@status_decorator
def clean_data(dataframe):
    """
    Here, we're using median as a central tendency. In a highly skewed dataset
    like this, there is a little difference between the median and mean.
    Also, it preserves the datatype of that column if it is a type of integer,
    for which using mean might introduce a float value
    """
    print("Cleaning data...",end="")
    if dataframe.isnull().values.any():
        dataframe.fillna(dataframe.median(),inplace=True)
        dataframe.dropna(inplace=True)

@status_decorator
def dim_reduction(dataframe,dim):
    print("Reducing dimensions to",dim,end="")
    global V
    meanframe = dataframe.mean()
    stdframe = dataframe.std(ddof=0,numeric_only=True)
    print(" from",len(meanframe.index),end="\n")
    features = meanframe.index
    for feature in features:
        dataframe['N'+feature] = (dataframe[feature] - meanframe[feature])/stdframe[feature]

    features = "N"+features
    covariance_matrix = dataframe[features].cov()
    eig_vals,eig_vecs = np.linalg.eig(covariance_matrix)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]

    V = eig_vecs[:,:dim]
    dataframe = pd.concat([dataframe,dataframe[features]@V],axis=1)
    print(dataframe)
    

def preprocess(dataframe):
    
    #column name count, has a name collision with an inbuilt function of pandas
    dataframe.rename(columns={"count":"counts"},inplace=True)

    #clean_data(dataframe,"Cleaning...")
    clean_data(dataframe)
    change_dtypes(dataframe)
    dim_reduction(dataframe,6)

