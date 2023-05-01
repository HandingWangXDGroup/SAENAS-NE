
import json
import os
import logging
import argparse
from evolution import ENAS
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from nasspace import Nasbench301, Nasbench201, Nasbench101
from gensim.models.doc2vec import Doc2Vec
from utils import merge_params

cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True

parser = argparse.ArgumentParser("nasbench")
parser.add_argument("--nasbench",choices=["101","201","301"],default="201")
parser.add_argument("--num_classes",type=int,default=10)
parser.add_argument("--layers",type=int,default=2)
parser.add_argument("--channels",type=int,default=32) #32
parser.add_argument("--nodes",type=int,default=4)
parser.add_argument("--init_lr",type=float,default=0.025)
parser.add_argument("--momentum",type=float,default=0.9)
parser.add_argument("--weight_decay",type=float,default=3e-4)
parser.add_argument("--epochs",type=int,default=150)
parser.add_argument("--root",type=str,default='/data/Fanliang/data')
parser.add_argument("--batch_size",type=int,default=256) #
parser.add_argument("--seed",type=int,default=0) # train


parser.add_argument("--pop_size",type=int,default=30) #m
parser.add_argument("--total_eval",type=int,default=300)
parser.add_argument("--p_c",type=float,default=0.5)
parser.add_argument("--p_m",type=float,default=0.05)

args = parser.parse_args()

args = merge_params(args)

print(args)
# graph2vec
g2v_model =  Doc2Vec.load("g2v_model/"+args.nasbench+"/doc2vec_model_dim128.model")

logging.basicConfig(level=logging.INFO,
                    filename='logs/nasbench-'+args.nasbench+'-'+args.dataset+'.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(levelname)s: %(message)s')

for seed in range(1):
    logging.info("seed:{}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.nasbench == "101":
        nasspace = Nasbench101(data_folder=args.nas_bench_dir)
    elif args.nasbench == "201":
        nasspace = Nasbench201("cifar10",args.nas_bench_dir)
    elif args.nasbench == "301":
        nasspace = Nasbench301()
    enas = ENAS(nasspace=nasspace,g2v_model=g2v_model,args=args)
    best_FS = enas.solve()