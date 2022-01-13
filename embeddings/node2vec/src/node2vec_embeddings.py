import argparse

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from node2vec import Node2Vec

__author__ = "Grover"

'''
The following code is a mixture of implementations from https://github.com/eliorc/node2vec
and https://github.com/aditya-grover/node2vec.
'''

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

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

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    
    nx_G = read_graph()
    node2vec = Node2Vec(
        graph=nx_G,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        workers=args.workers
    )

    model = node2vec.fit(seed=0)
    model.wv.save_word2vec_format(args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)
