import os
from pyexpat import model
from acc_predictor.meta_neural_net import MetaNeuralnet

from acc_predictor.mlp import predict
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from unicodedata import name
from utils import GraphDataset, archlist2archcode, build_train_sample, to_oneshot
import json
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
from gensim.models.doc2vec import Doc2Vec
from acc_predictor.factory import get_acc_predictor


# logging.basicConfig(level=logging.INFO,
#                     filename='log.log',
#                     filemode='w',
#                     format=
#                     '%(asctime)s - %(levelname)s: %(message)s')

# cudnn.enabled = True
# cudnn.benchmark = True
# cudnn.deterministic = True

# parser = argparse.ArgumentParser("nasbench-301")
# parser.add_argument("--classes",type=int,default=10)
# parser.add_argument("--layers",type=int,default=2)
# parser.add_argument("--channels",type=int,default=32) #32
# parser.add_argument("--nodes",type=int,default=4)
# parser.add_argument("--init_lr",type=float,default=0.025)
# parser.add_argument("--momentum",type=float,default=0.9)
# parser.add_argument("--weight_decay",type=float,default=3e-4)
# parser.add_argument("--epochs",type=int,default=150)
# parser.add_argument("--root",type=str,default='/data/Fanliang/data')
# # parser.add_argument("--batch_size",type=int,default=64) # train
# parser.add_argument("--batch_size",type=int,default=256) # kd
# parser.add_argument("--seed",type=int,default=0) # train
# # parser.add_argument("--seed",type=int,default=2020) # kd
# parser.add_argument("--ratio",type=float,default=0.8)
# parser.add_argument("--keep_prob",type=float,default=1.0)
# parser.add_argument("--drop_path_keep_prob",type=float,default=0.9)
# parser.add_argument('--use_aux_head', action='store_true', default=False)

# parser.add_argument("--n_kd_sample",type=int,default=20)

# parser.add_argument("--n_sample",type=int,default=3)
# parser.add_argument("--pop_size",type=int,default=30)
# parser.add_argument("--total_gen",type=int,default=100)

# parser.add_argument("--up_epochs",type=int,default=5)
# parser.add_argument("--up_lr",type=float,default=0.001)


# # args = parser.parse_args()
# # args.pth_file  = "supernet_1.pth"

# # random.seed(args.seed)
# # np.random.seed(args.seed)
# # torch.manual_seed(args.seed)
# # torch.cuda.manual_seed(args.seed)
# # torch.cuda.manual_seed_all(args.seed)

# searchSpace = Nasbench301()

# # # arch = searchSpace.get_cell().random_cell(nasbench=searchSpace.nasbench,random_encoding="adj")
# # # print(arch)

# # # adjs = []
# # # features = []

# # # arch2vec
# # # for _ in range(600000):
# # #     arch = searchSpace.get_cell().random_cell(searchSpace.nasbench,random_encoding="adj")
# # #     adj,feature = searchSpace.get_cell(arch).encode(predictor_encoding="gcn")
# # #     feature = to_oneshot(feature)
# # #     adjs.append(adj)
# # #     features.append(feature)

# # # adjs = torch.tensor(adjs).type(torch.float32)
# # # features  = torch.tensor(features).type(torch.float32)
# # # dataset = GraphDataset(adjs,features)
# # # model = Model(input_dim=16,hidden_dim=128,latent_dim=16,num_hops=5,num_mlp_layers=2,dropout=0.3).cuda()
# # # train_arch2vec(model,dataset)

# # # graph2vec
# g2v_model =  Doc2Vec.load("g2v_model_same/doc2vec_model_dim32.model")

# for num in [20,50,100,150,200]:
#     kds = []
#     for seed in range(20):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         model_pool = []
#         hash_visited = {}
#         XS,YS = [],[]
#         while len(hash_visited) < (num+200):
#             arch = Nasbench301.get_cell().random_cell(searchSpace.nasbench,random_encoding="adj")
#             hash_arch = searchSpace.get_hash(arch)
#             if hash_arch in hash_visited:
#                 continue
#             else:
#                 hash_visited[hash_arch]=1
#             F = 100-Nasbench301.get_cell(arch).get_val_loss(searchSpace.nasbench)
#             # gcn
#             # arch_dict = Nasbench301.get_cell(arch).encode(predictor_encoding="gcn",nasbench=searchSpace.nasbench)
#             # adj = torch.tensor([arch_dict[0]])
#             # adj = (adj+torch.transpose(adj,2,1)).type(torch.float32).cuda()
#             # features = torch.tensor([to_oneshot(arch_dict[1])]).type(torch.float32).cuda()
#             # arch2vec
#             # arch_code=model._encoder(features,adj)[0].cpu().detach().numpy()[0].reshape(-1)
            
#             # graph2vec
#             edges = []
#             features = {}
#             matrix,ops = Nasbench301.get_cell(arch).encode(predictor_encoding="gcn",nasbench=searchSpace.nasbench)
#             hash_info = str(searchSpace.get_hash(arch))
#             xs,ys = np.where(matrix==1)
#             xs = xs.tolist()
#             ys = ys.tolist()
#             for x,y in zip(xs,ys):
#                 edges.append([x,y])
#             for id in range(len(ops)):
#                 features[str(id)] = str(ops[id])
#             g = {"edges":edges,"features":features}
#             doc = featrue_extract_by_graph(g,name=hash_info)[0]
#             arch_code = g2v_model.infer_vector(doc)
#             # trunc_path
#             # arch_code = Nasbench301.get_cell(arch).encode(predictor_encoding='trunc_path',nasbench=searchSpace.nasbench)
#             # adj
#             # arch_code = Nasbench301.get_cell(arch).encode(predictor_encoding="adj",nasbench=searchSpace.nasbench)
#             model_pool.append((arch_code,F))
#             XS.append(arch_code)
#             YS.append(F)
        
#         ranknet = RankNet(32)
#         train_samples = build_train_sample(model_pool[:num])
#         ranknet.fit(*train_samples)
#         x_to_predict = [X[0] for X in model_pool[num:]]
#         scores = np.squeeze(ranknet.predict(x_to_predict).detach().cpu().numpy())
#         accs = [X[1] for X in model_pool[num:]]
#         kd = kendalltau(scores,accs)[0]
#         # train_X = np.array(XS[:num])
#         # train_Y = np.array(YS[:num])
#         # predictor = get_acc_predictor(model="rbf",inputs=train_X,targets=train_Y)

#         # meta_neuralnet = MetaNeuralnet()
#         # metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
#         #     'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
#         # meta_neuralnet.fit(train_X,train_Y,**metanet_params)
        
#         # test_X = np.array(XS[num:])
#         # test_Y = np.array(YS[num:])
#         # scores = np.squeeze(meta_neuralnet.predict(test_X))
#         # # scores = predictor.predict(test_X)
#         # kd = kendalltau(scores,test_Y)[0]
#         # kds.append(kd)
#         print("num:{} kd_test:{}".format(num,kd))
#         kds.append(kd)
#     print("kds_mean: {} kds_std: {}".format(np.mean(kds),np.std(kds)))



FS = []
FS_trace = []
for i in range(59):
    with open("FS_rank_same/seed-{}.json".format(i),'r') as fp:
        fs  = json.load(fp)
    FS.append(fs[-1])
    FS_trace.append(fs)
print(np.mean(FS))
print(np.std(FS))
print(FS)
