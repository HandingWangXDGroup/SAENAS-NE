import json
import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)
import time
import argparse
import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
from numpy.core.fromnumeric import argsort
from encoder.graph2vec import graph2vec
from operations import OPERATIONS_201
from nasspace import Nasbench301,Nasbench201




def load_graph_201():
    INPUT = 'input'
    OUTPUT = 'output'
    OPS = list(OPERATIONS_201.keys())
    OPS_INCLUSIVE = [INPUT, OUTPUT, *OPS]
    searchspace = Nasbench201()
    k=0
    for uid in range(15625):
        matrix = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]])
        edges = []
        features = {}
        arch_str = {"string":searchspace.nasbench.get_net_config(uid,dataset="cifar10-valid")['arch_str']}
        ops = ["input"]+searchspace.get_cell(arch_str).get_op_list()+["output"]
        if 'none' in ops:
            index = ops.index('none')
            matrix[index,:] = 0
            matrix[:,index] = 0

        xs,ys = np.where(matrix==1)
        xs = xs.tolist()
        ys = ys.tolist()
        for x,y in zip(xs,ys):
            edges.append([x,y])

        for id in range(len(ops)):
            features[str(id)] = str(OPS_INCLUSIVE.index(ops[id]))
        g = {"edges":edges,"features":features}
        with open("data/graphs_json/{}.json".format(uid),'w') as fp:
            json.dump(g,fp)
        k+=1
        print("{}-th arch has been writen".format(k))

def load_graph_301():
    searchspace = Nasbench301()
    k=0
    for uid in range(600000):
        edges = []
        features = {}
        arch = Nasbench301.get_cell().random_cell(searchspace.nasbench,random_encoding="adj")
        matrix,ops = Nasbench301.get_cell(arch).encode(predictor_encoding="gcn")

        xs,ys = np.where(matrix==1)
        xs = xs.tolist()
        ys = ys.tolist()
        for x,y in zip(xs,ys):
            edges.append([x,y])
        for id in range(len(ops)):
            features[str(id)] = str(ops[id])
        g = {"edges":edges,"features":features}
        with open("data/graphs_json_same/{}.json".format(uid),'w') as fp:
            json.dump(g,fp)
        k+=1
        print("{}-th arch has been writen".format(k))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")
    parser.add_argument("--nasbench",choices=["101","201","301"],default="201")
    parser.add_argument("--input-path",
                        nargs="?",
                        default="./data/graphs_json_same",
                        help="Input folder with jsons.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="./data/feature/features.csv",
                        help="Embeddings path.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=32,
                        help="Number of dimensions. Default is 128.")

    parser.add_argument("--workers",
                        type=int,
                        default=32,
                        help="Number of workers. Default is 4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=150,
                        help="Number of epochs. Default is 10.")

    parser.add_argument("--min-count",
                        type=int,
                        default=5,
                        help="Minimal structural feature count. Default is 5.")

    parser.add_argument("--wl-iterations",
                        type=int,
                        default=2,
                        help="Number of Weisfeiler-Lehman iterations. Default is 2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
                        help="Initial learning rate. Default is 0.025.")

    parser.add_argument("--down-sampling",
                        type=float,
                        default=0.0001,
                        help="Down sampling rate of features. Default is 0.0001.")
    args = parser.parse_args()
    if args.nasbench=="201":
        load_graph_201()
    elif args.nasbench=="301":
        load_graph_301()
    graph2vec(args)