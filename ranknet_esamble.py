import warnings
warnings.filterwarnings('ignore')
from torch.optim import optimizer
from encoder.arch2vec import Model
import logging
from random import shuffle
from utils import RankDataset
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context
from tqdm import tqdm
from multiprocessing import Pool
import torch.multiprocessing as mp



class RankModel(nn.Module):
    def __init__(self, num_feature):
        super(RankModel, self).__init__()

        self.features = nn.Sequential(
            nn.Linear( num_feature, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU6(inplace=True),
            nn.Linear(32,1)
        )
        self.P_ij = nn.Sigmoid()

    def forward(self, x1,x2):
        x_i = self.features(x1)
        x_j = self.features(x2)
        S_ij = torch.add(x_i,-x_j)
        return self.P_ij(S_ij)
    
    def predict(self, input_):
        s = self.features(input_)
        return s

class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc,self).__init__()
    
    def forward(self,s_diff,S_ij):
        loss = torch.mean((1-S_ij)*s_diff/2. + torch.log(1+torch.exp(-s_diff)))
        return loss

def tain_OneModel(model,X1,X2,Y,total_epoch=50,batch_size=32):
    model.train()
    
    train_set = RankDataset(X1,X2,Y)
    loss_func = nn.BCELoss().cuda()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2
                              ,multiprocessing_context=get_context('loky'))
    for epoch in range(total_epoch):
        for step,(x1,x2,y) in enumerate(train_loader):
            x1,x2,y = x1.cuda(),x2.cuda(),y.cuda()
            optimizer.zero_grad()
            s_diff = model(x1,x2)
            S_ij = y
            S_ij = S_ij.view(-1,1)
            loss = loss_func(s_diff,S_ij)
            loss.backward()
            optimizer.step()
    del train_loader
    del train_set
    torch.cuda.empty_cache()
    return model


class RankNet(object):
    def __init__(self,num_feature,model=None,n_model=5):
        self.num_feature = num_feature
        self.n_model=n_model
        self.models=[]
        for _ in range(n_model):
            self.models.append(RankModel(num_feature).cuda())
        else:
            self.model=model
    
    def fit(self,X1,X2,Y,total_epoch=50,batch_size=32,optimizerAlgorithm="Adam"):
        XS1,XS2,YS = [],[],[]
        for _ in range(self.n_model):
            X1 = torch.tensor(X1).type(torch.float32)
            X2 = torch.tensor(X2).type(torch.float32)
            Y = torch.tensor(Y).type(torch.float32)
            XS1.append(X1)
            XS2.append(X2)
            YS.append(Y)
        self.models = Parallel(n_jobs=self.n_model,backend='loky')(delayed(tain_OneModel)(model,X1,X2,Y,total_epoch,batch_size) for X1,X2,Y,model in tqdm(zip(XS1,XS2,YS,self.models)))
                    

    def save_state(self,gen):
        torch.save(self.model.state_dict(),"rankpths/ranknet_"+str(gen)+".pth")

    # def set_optimizer(self,model,optimizerAlgorithm):
    #     if optimizerAlgorithm == "Adam":
    #         optimizer = optim.Adam(model.parameters(),lr=1e-3)
    #     elif optimizerAlgorithm == "SGD":
    #         optimizer = optim.SGD(model.parameters(),lr=1e-3)
    #     elif optimizerAlgorithm == "AdaGrad":
    #         optimizer = optim.Adagrad(model.parameters())
    #     return optimizer
    
    def load_state(self,pth_file=None):
        if pth_file is not None:
            self.model.load_state_dict(torch.load(pth_file))
        
    def predict(self,X):
        X = torch.tensor(X).type(torch.float32).cuda()
        scores = np.full((self.n_model,X.shape[0]),0.)
        k=0
        for model in self.models:
            score = model.predict(X)
            scores[k]=np.squeeze(score.cpu().detach().numpy())
            k+=1
        scores_idxs = np.argsort(scores)
        scores_rank = np.full(scores_idxs.shape,0)
        for i in range(scores_rank.shape[0]):
            scores_rank[i][scores_idxs[i]] = np.arange(scores_rank.shape[1])
        # logging.info("scores_rank:{}".format(scores_rank))
        pred = np.mean(scores_rank,axis=0)
        uncertainty = np.std(scores_rank,axis=0)
        return pred,uncertainty