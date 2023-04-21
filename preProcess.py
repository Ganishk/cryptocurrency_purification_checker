import pandas as pd, numpy as np

def change_dtypes(df):
    df.address  = df.address.astype('object')
    df.year     = df.year.astype(np.int64)
    df.day      = df.day.astype(np.int64)
    df.length   = df.length.astype(np.int64)
    df.weight   = df.weight.astype(np.float64)
    df.counts    = df.counts.astype(np.int64)
    df.looped   = df.looped.astype(np.int64)
    df.neighbors= df.neighbors.astype(np.int64)
    df.income   = df.income.astype(np.int64)
    df.label    = pd.Categorical(df.label)

def clean_data(dataframe):
    """
    Here, we're using median as a central tendency. In a highly skewed dataset
    like this, there is a little difference between the median and mean.
    Also, it preserves the datatype of that column if it is a type of integer,
    for which using mean might introduce a float value
    """
    if not dataframe.isnull().values.any():
        return
    dataframe.fillna(dataframe.median(),inplace=True)
    dataframe.dropna(inplace=True)

def dim_reduction(dataframe):
    pass

def preprocess(dataframe):
    
    #column name count, has a name collision with an inbuilt function of pandas
    dataframe.rename(columns={"count":"counts"},inplace=True)

    clean_data(dataframe)
    change_dtypes(dataframe)
