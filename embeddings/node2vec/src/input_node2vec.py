import pandas as pd
from tqdm import tqdm

import pickle

__author__ = "Grover"


# Covert string to the proper set format
def get_set(x):                 
    x = x[1:-1]
    return set(x.split(','))

if __name__ == "__main__":
    
    df = pd.read_csv('../data/string_locs2.csv')

    df['locations'] = df['locations'].apply(lambda x: get_set(x))

    print(df.shape)

    ids = list()
    for i in df.index:
        l = df['locations'][i]
        if len(l) != 1:                 # Remove rows that have more than 1 locations
            ids.append(i)

    df = df.drop(ids)
    print(df.shape)

    # df = df.sample(frac=1, random_state=7)
    df = df.iloc[:200000, :]             # Pick first 200,000 data-points

    # prot2idx = dict()
    # idx2prot = dict()
    # i = 0
    # for idx in tqdm(df.index):
    #     p1 = df['protein1'][idx]
    #     p2 = df['protein2'][idx]
    #     try:
    #         prot_id = prot2idx[p1]
    #     except:
    #         prot2idx[p1] = i
    #         idx2prot[i] = p1
    #         i += 1
    #     try:
    #         prot_id = prot2idx[p2]
    #     except:
    #         prot2idx[p2] = i
    #         idx2prot[i] = p2
    #         i += 1

    # df['protein1'] = df['protein1'].apply(lambda x: prot2idx[x])
    # df['protein2'] = df['protein2'].apply(lambda x: prot2idx[x])

    # with open('./AttentionWalk/idx2prot.pickle', 'wb') as f:
    #     pickle.dump(idx2prot, f)

    df2 = df[['protein1', 'protein2', 'locations']]
    max_score = max(df['combined_score'].to_list())
    df['combined_score'] = df['combined_score'].apply(lambda x: ((x/max_score)))
    # df = df[['protein1', 'protein2']]
    df = df[['protein1', 'protein2', 'combined_score']]
    print(df.shape)

    # Save the files. The helper file will help us to match the locations later on.
    df2.to_csv('../data/graph_helper.csv', index=None)    
    df.to_csv('../data/selected.edgelist', sep=' ', index=None, header=False)  
        
    # df2.to_csv('./AttentionWalk/input/prot_helper.csv', index=None)
    # df.to_csv('./AttentionWalk/input/protgraph.csv', index=None)