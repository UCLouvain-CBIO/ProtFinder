import pandas as pd
from tqdm import tqdm

import pickle

__author__ = "Grover"


# Covert string to the proper set format
# def get_set(x):                 
#     x = x[1:-1].strip()
#     x = x.replace("'", "")
#     return set(x.split(','))

if __name__ == "__main__":
    
    df = pd.read_csv('../data/string_bioplex.csv')
    df = df[df.reliability1.notnull()]
    df = df[df.reliability2.notnull()]
    # df = df.loc[df['source'] == 'string']

    df = df.sample(frac=1, random_state=7)

    df = df.iloc[:150000, :]                # pick first 64,861 datapoints (max # of bioplex points)

    df2 = df[['protein1', 'protein2', 'reliability1', 'reliability2', 'locations']]
    df = df[['protein1', 'protein2', 'combined_score']]

    print(df.shape) 

    df2.to_csv('../data/combined_more_new_helper.csv', index=None)    
    df.to_csv('../data/combined_more_new.edgelist', sep=' ', index=None, header=False)