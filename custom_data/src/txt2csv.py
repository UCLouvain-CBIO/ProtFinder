import pandas as pd

txt = pd.read_csv('../data/9606.protein.links.detailed.v11.0.txt', delimiter=' ')
txt.to_csv('../data/string_human.csv', index=None)
