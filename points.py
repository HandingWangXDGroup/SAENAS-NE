import json
from sys import hash_info
from unicodedata import name
from utils import GraphDataset, archlist2archcode, build_train_sample, to_oneshot
import os
import logging
import argparse
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from nasspace import Nasbench301
from common import build_oneshot,build_data_loader,build_criterion, train_model,train_oneshot,valid_oneshot
from scipy.stats import kendalltau,pearsonr,spearmanr
from ranknet import RankNet
from encoder.arch2vec import Model
from encoder.trainer import train_arch2vec
from encoder.graph2vec import featrue_extract_by_graph

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
parser.add_argument("--pop_size",type=int,default=30)
parser.add_argument("--total_gen",type=int,default=100)

parser.add_argument("--up_epochs",type=int,default=5)
parser.add_argument("--up_lr",type=float,default=0.001)


args = parser.parse_args()
args.pth_file  = "supernet_1.pth"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

searchSpace = Nasbench301()

adjs = []
features = []

# arch2vec
# for _ in range(10000):
#     arch = searchSpace.get_cell().random_cell(searchSpace.nasbench,random_encoding="adj")
#     adj,feature = searchSpace.get_cell(arch).encode(predictor_encoding="gcn")
#     feature = to_oneshot(feature)
#     adjs.append(adj)
#     features.append(feature)

# adjs = torch.tensor(adjs).type(torch.float32)
# features  = torch.tensor(features).type(torch.float32)
# dataset = GraphDataset(adjs,features)
# model = Model(input_dim=9,hidden_dim=128,latent_dim=16,num_hops=5,num_mlp_layers=2,dropout=0.3).cuda()
# train_arch2vec(model,dataset)

# graph2vec
g2v_model =  torch.load("graph2vec.pth")

kds = []
sps = []
pss = []
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model_pool = []
archs = []
accs = []
for kid in range(10000):
    arch = Nasbench301.get_cell().random_cell(searchSpace.nasbench,random_encoding="adj")
    F = 100-Nasbench301.get_cell(arch).get_val_loss(searchSpace.nasbench)
    # gcn
    # arch_dict = Nasbench301.get_cell(arch).encode(predictor_encoding="gcn",nasbench=searchSpace.nasbench)
    # adj = torch.tensor([arch_dict[0]])
    # adj = (adj+torch.transpose(adj,2,1)).type(torch.float32).cuda()
    # features = torch.tensor([to_oneshot(arch_dict[1])]).type(torch.float32).cuda()
    # # arch2vec
    # arch_code=model._encoder(features,adj)[0].cpu().detach().numpy()[0].reshape(-1)

    # graph2vec
    edges = []
    features = {}
    matrix,ops = Nasbench301.get_cell(arch).encode(predictor_encoding="gcn",nasbench=searchSpace.nasbench)
    hash_info = str(searchSpace.get_hash(arch))
    xs,ys = np.where(matrix==1)
    xs = xs.tolist()
    ys = ys.tolist()
    for x,y in zip(xs,ys):
        edges.append([x,y])
    for id in range(len(ops)):
        features[str(id)] = str(ops[id])
    g = {"edges":edges,"features":features}
    doc = featrue_extract_by_graph(g,name=hash_info)[0]
    arch_code = g2v_model.infer_vector(doc)
    # trunc_path
    # arch_code = Nasbench301.get_cell(arch).encode(predictor_encoding='trunc_path',nasbench=searchSpace.nasbench)
    # model_pool.append((arch_code,F))
    archs.append(arch_code.tolist())
    accs.append(F)
    print("{}-th is ok".format(kid+1))
with open("points_arch.json",'w') as fp:
    json.dump(archs,fp)
with open("points_acc.json",'w') as fp:
    json.dump(accs,fp)