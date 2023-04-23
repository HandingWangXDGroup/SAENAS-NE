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
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight,a=1)
                nn.init.constant_(m.bias,0)

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


class RankNet(object):
    def __init__(self,num_feature,model=None):
        self.num_feature = num_feature
        if model == None:
            self.model=RankModel(num_feature).cuda()
        else:
            self.model=model

    
    def fit(self,X1,X2,Y,total_epoch=50,batch_size=512,optimizerAlgorithm="Adam"):
        X1 = torch.tensor(X1).type(torch.float32)
        X2 = torch.tensor(X2).type(torch.float32)
        Y = torch.tensor(Y).type(torch.float32)
        train_set = RankDataset(X1,X2,Y)
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)
        self.set_loss()
        self.set_optimizer(optimizerAlgorithm)
        self.model.train()
        for epoch in range(total_epoch):
            for step,(x1,x2,y) in enumerate(train_loader):
                x1,x2,y = x1.cuda(),x2.cuda(),y.cuda()
                self.optimizer.zero_grad()
                s_diff = self.model(x1,x2)
                S_ij = y
                S_ij = S_ij.view(-1,1)
                loss = self.loss_func(s_diff,S_ij)
                loss.backward()
                self.optimizer.step()
            # logging.info("epoch-{} loss:{}".format(epoch+1,loss.item()))

    def save_state(self,gen):
        torch.save(self.model.state_dict(),"rankpths/ranknet_"+str(gen)+".pth")

    def set_optimizer(self,optimizerAlgorithm):
        if optimizerAlgorithm == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3)
        elif optimizerAlgorithm == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),lr=1e-3)
        elif optimizerAlgorithm == "AdaGrad":
            self.optimizer = optim.Adagrad(self.model.parameters())
    
    def set_loss(self):
        self.loss_func = nn.BCELoss().cuda()
    
    def load_state(self,pth_file=None):
        if pth_file is not None:
            self.model.load_state_dict(torch.load(pth_file))
        
    def predict(self,X):
        X = torch.tensor(X).type(torch.float32).cuda()
        score = self.model.predict(X)
        return score

