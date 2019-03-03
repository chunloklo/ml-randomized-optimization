import pandas as pd
def convert_numeric(df, arr):
    for val in arr:
        df[val] = pd.Categorical(df[val])
        df[val] = df[val].cat.codes
    return df

def XySplit(df, label_col):
    X = df.drop([label_col], axis=1)
    y = df[label_col]
    return X, y