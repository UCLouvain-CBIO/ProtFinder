import pandas as pd
from tqdm import tqdm

import pickle

__author__ = "Grover"


# Covert string to the proper set format
def get_set(x):                 
    x = x[1:-1]
    return set(x.split(','))

if __name__ == "__main__":
    
    df = pd.read_csv('../data/string_bioplex.csv')

    df['locations'] = df['locations'].apply(lambda x: get_set(x))
    print(df.shape)

    df2 = df[['protein1', 'reliability1', 'protein2', 'reliability2', 'locations']]
    max_score = max(df['combined_score'].to_list())
    df['combined_score'] = df['combined_score'].apply(lambda x: ((x/max_score)))
    # df = df[['protein1', 'protein2']]
    df = df[['protein1', 'protein2', 'combined_score']]
    print(df.shape)

    # Save the files. The helper file will help us to match the locations later on.
    df2.to_csv('../data/graph_helper_all.csv', index=None)    
    df.to_csv('../data/input_all.edgelist', sep=' ', index=None, header=False)  
