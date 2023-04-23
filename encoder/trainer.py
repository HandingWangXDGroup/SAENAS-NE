import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def train_arch2vec(model,dataset,batch_size=32,epochs=20):
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)
    loss_fn  = nn.BCELoss().cuda()
    model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    for epoch in range(epochs):
        for step,(adjs,features) in enumerate(dataloader):
            optimizer.zero_grad()
            adjs,features = adjs.cuda(),features.cuda()
            adjs = adjs + adjs.triu(1).transpose(-1, -2)
            ops_recon, adjs_recon, mu, logvar = model(features,adjs)
            adjs_recon,ops_recon = prep_reverse(adjs_recon,ops_recon)
            adjs,features = prep_reverse(adjs,features)
            loss = VAEReconstructed_Loss(loss_ops=F.mse_loss,loss_adj=F.mse_loss)((ops_recon, adjs_recon), (features, adjs), mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step%1000==0:
                logging.info('epoch {}: batch {} : loss: {:.5f}'.format(epoch, step+1, loss.item()))
    torch.save(model.state_dict(),"arch2vec.pth")

class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD

def prep_reverse(A, H):
    return A.triu(1), H
