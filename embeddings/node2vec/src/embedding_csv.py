
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import pickle
from tqdm import tqdm


# %%
df = pd.read_csv('../data/all.emb', delimiter=' ', header=None)

hidden_dim = df.shape[1] - 1

columns = ['protein']
for i in range(hidden_dim):
    columns.append(('dim'+str(i+1)))

df.columns = columns

proteins = set(df['protein'].to_list())
len(proteins)


# %%
# Covert string to the proper set format, since each location is of the form {'location_name'}
def get_set(x):
    x = x[1:-1]
    x = x.replace("'", "")
    x = x.replace('"', '"').strip()
    x = x.split(',')
    x = set([i.strip() for i in x])
    return x


# %%
omit_locs = ['Rods & Rings', 'Aggresome', 'Microtubule ends', 'Cleavage furrow']

def get_loc_values(df, info_retain=1.0):
    
    # Compute a dictionary of the form {protein_id: {location: count}, ...}
    prot2loc_c = dict()
    for i in tqdm(df.index):
        p1 = df['protein1'][i]
        p2 = df['protein2'][i]
        l = get_set(df['locations'][i])

        try:
            prot2loc_c[p1]
        except:
            prot2loc_c[p1] = dict()
        try:
            prot2loc_c[p2]
        except:
            prot2loc_c[p2] = dict()

        for loc in l:
            if loc not in omit_locs:
                try:
                    prot2loc_c[p1][loc] += 1
                except:
                    prot2loc_c[p1][loc] = 1
            
                try:
                    prot2loc_c[p2][loc] += 1
                except:
                    prot2loc_c[p2][loc] = 1

    # Pick locations corresponding to a protein that hold more information than info_retain
    for i in tqdm(prot2loc_c.keys()):
        counts = prot2loc_c[i].values()
        total_c = sum(counts)
        counts = [c/total_c for c in counts]
        norm_counts = dict(zip(prot2loc_c[i].keys(), counts))
        norm_counts = {
            k: v for k, v in sorted(norm_counts.items(), key=lambda x: x[1], reverse=True)
            }

        cumsum = 0
        temp = dict()
        for key,val in norm_counts.items():
            if cumsum > info_retain:
                break
            cumsum += val
            temp[key] = val

        prot2loc_c[i] = set(temp.keys())

    return prot2loc_c


# %%
data = pd.read_csv('../data/graph_helper_all.csv')

data = data[data.reliability1.notnull()]
data = data[data.reliability2.notnull()]

prot2loc = get_loc_values(data, info_retain=0.6)

# prot2loc = dict()
# locs = set()
# drop_l = list()
# for i in data.index:

#     l = get_set(data['locations'][i])
#     for x in l:
#         locs.add(x)
            
#     p1 = data['protein1'][i]  
#     try:
#         prot2loc[p1] = prot2loc[p1].union(l)
#     except:
#         prot2loc[p1] = l

#     p2 = data['protein2'][i]
#     try:
#         prot2loc[p2] = prot2loc[p2].union(l)
#     except:
#         prot2loc[p2] = l
p = next(iter(prot2loc))
print(p, prot2loc[p])
# list(prot2loc.values())[0]


# %%
# locs = list(locs)
# loc2id = {k: v for v, k in enumerate(locs)}

# with open('../data/loc2id_string_new.pkl','wb') as f:
#         pickle.dump(loc2id, f)

# for key,value in loc2id.items():
#         print(key, value)


# %%
df.head()


# %%
df_locs = list()
for i in df.index:
    try:
        loc = prot2loc[df['protein'][i]]
    except:
        loc = None
    df_locs.append(loc)
        
df['locations'] = df_locs

df_null = df[df.locations.isnull()]

df.dropna(inplace=True)

print(df.shape, df_null.shape)
print(df_null.head(1))

# %%
cols = list(df.columns)
cols.remove('locations')
df_null = df_null[cols]

cols.remove('protein')
cols2 = ['locations']+cols

df = df[cols2].sample(frac=1, random_state=42)

n_prots = df.shape[0]
alpha = 0.85
n_train = int(alpha * n_prots)
df_train = df.iloc[:n_train, :]
df_test = df.iloc[n_train:, :]

print(df_train.shape, df_test.shape, df_null.shape)


# %%
df_train.to_csv('../data/train_all_0.6.csv', index=None)
df_test.to_csv('../data/test_all_0.6.csv', index=None)
df_null.to_csv('../data/infer_all_0.6.csv', index=None)


# %%



