import logging
import random
import numpy as np
import torch
from torch.functional import split
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model_search import NASWSNetworkCIFAR
from utils import AvgrageMeter,accuracy
from nasspace import Nasbench301

def build_oneshot(args):
    model = NASWSNetworkCIFAR(args.classes, args.layers, args.nodes, args.channels,args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.init_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,0.001,-1)
    return model,optimizer,lr_scheduler

def build_data_loader(args):
    transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
        [0.24703233, 0.24348505, 0.26158768])
    ])

    train_data = datasets.CIFAR10(root=args.root,train=True,transform=transform,download=False)
    test_data = datasets.CIFAR10(root=args.root,train=False,transform=transform,download=False)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(num_train*args.ratio)
    np.random.shuffle(indices)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                            pin_memory=True,num_workers=8) 
    eval_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
                                            pin_memory=True,num_workers=8)
    
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,shuffle=False,num_workers=8)
    return train_loader,eval_loader,test_loader

def build_criterion():
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    return train_criterion,eval_criterion

def train_oneshot(model,train_loader,optimizer,train_ceriterion,lr_scheduler,epochs,nasbench:Nasbench301):
    model.train()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    global_step = 0
    for epoch in range(epochs):
        objs.reset()
        top1.reset()
        top5.reset()
        for step,(input,target) in enumerate(train_loader):
            input,target = input.cuda(),target.cuda()
            optimizer.zero_grad()
            arch = nasbench.random_arch()
            output,_ = model(input,arch,step=global_step)
            global_step+=1
            loss = train_ceriterion(output,target)
            prec1,prec5 = accuracy(output,target,topk=(1,5))
            loss.backward()
            optimizer.step()
            n = target.size(0)
            objs.update(loss.item(),n)
            top1.update(prec1.item(),n)
            top5.update(prec5.item(),n)
        lr_scheduler.step()
        logging.info("epoch:{} loss:{} top1:{} top5:{}".format(epoch+1,objs.avg,top1.avg,top5.avg))
        if (epoch+1)%30==0:
            torch.save(model.state_dict(),"spospths/"+str(epoch+1)+".pth")
            

def valid_oneshot(model,test_loader,test_ceritrion,arch):
    model.eval()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    with torch.no_grad():
        for step,(input,target) in enumerate(test_loader):
            input,target = input.cuda(),target.cuda()
            output,_ = model(input,arch,bn_train=True)
            loss = test_ceritrion(output,target)
            n = input.shape[0]
            prec1,prec5 = accuracy(output,target,topk=(1,5))
            objs.update(loss.item(),n)
            top1.update(prec1.item(),n)
            top5.update(prec5.item(),n)
    return top1.avg

def train_model(model,train_loader,optimizer,train_ceriterion,arch,epochs):
    model.train()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    global_step = 0
    for epoch in range(epochs):
        objs.reset()
        top1.reset()
        top5.reset()
        for step,(input,target) in enumerate(train_loader):
            input,target = input.cuda(),target.cuda()
            optimizer.zero_grad()
            output,_ = model(input,arch,step=global_step)
            global_step+=1
            loss = train_ceriterion(output,target)
            prec1,prec5 = accuracy(output,target,topk=(1,5))
            loss.backward()
            optimizer.step()
            n = target.size(0)
            objs.update(loss.item(),n)
            top1.update(prec1.item(),n)
            top5.update(prec5.item(),n)
        logging.info("epoch:{} loss:{} top1:{} top5:{}".format(epoch+1,objs.avg,top1.avg,top5.avg))

