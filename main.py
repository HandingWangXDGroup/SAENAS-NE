
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
import argparse
from evolution import ENAS
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from nasspace import Nasbench301
from gensim.models.doc2vec import Doc2Vec

logging.basicConfig(level=logging.INFO,
                    filename='log.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(levelname)s: %(message)s')

cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True

parser = argparse.ArgumentParser("nasbench-301")
parser.add_argument("--classes",type=int,default=10)
parser.add_argument("--layers",type=int,default=2)
parser.add_argument("--channels",type=int,default=32) #32
parser.add_argument("--nodes",type=int,default=4)
parser.add_argument("--init_lr",type=float,default=0.025)
parser.add_argument("--momentum",type=float,default=0.9)
parser.add_argument("--weight_decay",type=float,default=3e-4)
parser.add_argument("--epochs",type=int,default=150)
parser.add_argument("--root",type=str,default='/data/Fanliang/data')
# parser.add_argument("--batch_size",type=int,default=64) # train
parser.add_argument("--batch_size",type=int,default=256) # kd
parser.add_argument("--seed",type=int,default=0) # train
# parser.add_argument("--seed",type=int,default=2020) # kd
parser.add_argument("--ratio",type=float,default=0.8)
parser.add_argument("--keep_prob",type=float,default=1.0)
parser.add_argument("--drop_path_keep_prob",type=float,default=0.9)
parser.add_argument('--use_aux_head', action='store_true', default=False)

parser.add_argument("--n_kd_sample",type=int,default=20)

parser.add_argument("--n_sample",type=int,default=3)
parser.add_argument("--pop_size",type=int,default=30) #m
parser.add_argument("--total_gen",type=int,default=80) #m
parser.add_argument("--total_eval",type=int,default=300)
parser.add_argument("--p_c",type=float,default=0.5)
parser.add_argument("--p_m",type=float,default=0.05)
parser.add_argument("--pool_limit",type=int,default=50)

parser.add_argument("--up_epochs",type=int,default=5)
parser.add_argument("--up_lr",type=float,default=0.001)

args = parser.parse_args()
# graph2vec
g2v_model =  Doc2Vec.load("g2v_model_same/doc2vec_model_dim32.model")


for seed in range(2,200):
    logging.info("seed:{}".format(seed))
    nasspace = Nasbench301()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    enas = ENAS(nasspace=nasspace,g2v_model=g2v_model,args=args)
    best_FS,diversity = enas.solve()
    with open("FS_rank_same/seed-{}.json".format(seed),'w') as fp:
        json.dump(best_FS,fp)
    with open("diversity_same/seed-{}.json".format(seed),'w') as fp:
        json.dump(diversity,fp)
    

# preds0 = []
# preds1 = []
# trues = []
# for i in range(20):
#     arch = nasbench.random_arch()
#     true = nasbench.get_val_acc(arch)
#     model.load_state_dict(torch.load(args.pth_file))
#     pred0 = valid_oneshot(model,eval_loader,eval_criterion,arch)
#     up_optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=args.up_lr,
#         momentum=args.momentum,
#         weight_decay=args.weight_decay
#     )
#     train_model(model,train_loader,up_optimizer,train_criterion,arch,args.up_epochs)
#     pred1=valid_oneshot(model,eval_loader,eval_criterion,arch)
#     logging.info("i-{} pred0-{} pred1-{} true-{}".format(i+1,pred0,pred1,true))
#     preds0.append(pred0)
#     preds1.append(pred1)
#     trues.append(true)

# kd0,p_value0 = kendalltau(preds0,trues)
# kd1,p_value1 = kendalltau(preds1,trues)
# logging.info("kd0-{} kd1-{}".format(kd0,kd1))