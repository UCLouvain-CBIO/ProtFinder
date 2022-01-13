import argparse
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from networkx.readwrite.json_graph import node_link_data

from sklearn.preprocessing import MultiLabelBinarizer


__author__ = "Grover"


classes = [
	'Actin filaments',
	'Cell Junctions',
	'Centriolar satellite',
	'Centrosome',
	'Cytokinetic bridge',
	'Cytoplasmic bodies',
	'Cytosol',
	'Endoplasmic reticulum',
	'Endosomes',
	'Focal adhesion sites',
	'Golgi apparatus',
	'Intermediate filaments',
	'Lipid droplets',
	'Lysosomes',
	'Microtubules',
	'Midbody',
	'Midbody ring',
	'Mitochondria',
	'Mitotic spindle',
	'Nuclear bodies',
	'Nuclear membrane',
	'Nuclear speckles',
	'Nucleoli',
	'Nucleoli fibrillar center',
	'Nucleoplasm',
	'Peroxisomes',
	'Plasma membrane',
	'Vesicles'
]

omit_locs = ['Rods & Rings', 'Aggresome', 'Microtubule ends', 'Cleavage furrow']

def get_list(x):
	x = x[1:-1]
	x = x.replace('"', '')
	x = x.replace("'", "").strip()
	x = x.split(',')
	x = [i.strip() for i in x]
	return np.array(x)


def parse_args():
	'''
	Parses the arguments.
	'''
	parser = argparse.ArgumentParser(description="Run GraphSage.")

	parser.add_argument('--input', nargs='?', default='../data/selected.edgelist',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='../data/select_embed.emb',
						help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
						help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
					  help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
						help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=str, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	print('Graph is built.')

	return G

def get_id(data):
	'''
	Returns a mapping between the protein id and a numeric id
	'''
	ids = set()
	p2id = dict()
	for i, node in enumerate(data['nodes']):
		pid = node['id']
		l1 = len(ids)
		ids.add(pid)
		l2 = len(ids)
		if l1 == l2:
			continue
		else:
			p2id[pid] = i
	return p2id

def multihot(loc):
	loc = list(loc)
	enc = MultiLabelBinarizer(classes=classes)
	mat = enc.fit_transform([loc])
	return mat

def get_loc_values(df, info_retain=1.0):
	
	# Compute a dictionary of the form {protein_id: {location: count}, ...}
	prot2loc_c = dict()
	for i in tqdm(df.index):
		p1 = df['protein1'][i]
		p2 = df['protein2'][i]
		l = get_list(df['locations'][i])

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
	
	prot2loc_c = {k: multihot(v) for k, v in prot2loc_c.items()}

	return prot2loc_c

def get_locations():
	file_l = (args.input).split('/')[:-1]
	path = '/'.join(file_l)
	fname = ((args.input).split('/')[-1]).split('.')[0] + '_helper.csv'
	df = pd.read_csv(path+'/'+fname)

	prot2loc = get_loc_values(df, info_retain=1) 

	return prot2loc

def check(id1, id2, data):
	val1, val2 = -1, -1
	for i,_ in enumerate(data['nodes']):
		if id1 == data['nodes'][i]['id']:
			if data['nodes'][i]['test']:
				val1 = 2
			elif data['nodes'][i]['val']:
				val1 = 1
			else:
				val1 = 0
		elif id2 == data['nodes'][i]['id']:
			if data['nodes'][i]['test']:
				val2 = 2
			elif data['nodes'][i]['val']:
				val2 = 1
			else:
				val2 = 0

		if val1 != -1 and val2 != -1:
			return max([val1, val2])
		
		

def process_data(data):
	'''
	Prepares the data (dict) in the required format for GraphSAGE
	'''
	del data['multigraph']

	p2id = get_id(data)
	p2loc = get_locations()

	n = len(data['nodes'])
	val_split = int(0.1*n)
	test_split = int(0.1*n)
	for i,_ in enumerate(data['nodes']):
		if i < val_split:
			data['nodes'][i]['test'] = False
			data['nodes'][i]['val'] = True
		elif (i - val_split) < test_split:
			data['nodes'][i]['val'] = False
			data['nodes'][i]['test'] = True
		else:
			data['nodes'][i]['val'] = False
			data['nodes'][i]['test'] = False
		data['nodes'][i]['label'] = (p2loc[data['nodes'][i]['id']]).tolist()[0]
		data['nodes'][i]['id'] = p2id[data['nodes'][i]['id']]
		data['nodes'][i]['feature'] = (np.random.normal(size=128)).tolist()
	
	
	for i,_ in enumerate(data['links']):
		
		data['links'][i]['target'] = p2id[data['links'][i]['target']]
		data['links'][i]['source'] = p2id[data['links'][i]['source']]

		id1 = data['links'][i]['source']
		id2 = data['links'][i]['target']
		idx = check(id1, id2, data)

		if idx == 0:	
			data['links'][i]['test_removed'] = False
			data['links'][i]['train_removed'] = False
		elif idx == 1:
			data['links'][i]['test_removed'] = False
			data['links'][i]['train_removed'] = True
		elif idx == 2:
			data['links'][i]['test_removed'] = True
			data['links'][i]['train_removed'] = True


	return data

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	
	nx_G = read_graph()
	data = node_link_data(nx_G)
	data = process_data(data)

	map_id = dict()
	class_id = dict()
	for i in range(len(data['nodes'])):
		map_id[str(i)] = i
		class_id[str(i)] = data['nodes'][i]['label']

	with open('./data/combined-G.json', 'w') as f:
		json.dump(data, f)
	with open('./data/combined-class_map.json','w') as f:
		json.dump(class_id, f)
	with open('./data/combined-id_map.json','w') as f:
		json.dump(map_id, f)


if __name__ == "__main__":
	args = parse_args()
	main(args)