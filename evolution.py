
import copy
from nondo_sort import NonDominatedSorting
import numpy as np
from population import Population
from ranknet import RankNet

from utils import  build_train_sample
from individual import Individual

from nasspace import  Nasbench301

import logging
import random
from operations import OPERATIONS

operations = list(OPERATIONS.keys())
n_operations = len(operations)

def tournament_select(pop,n_sample=2):
    pop_size  = len(pop)
    idxs = random.sample(range(pop_size),k=n_sample)
    scores_selected = [pop[i].score for i in idxs]
    id  = idxs[np.argmax(scores_selected)]
    return pop[id]

def crossover_ind(p1,p2,nasbench:Nasbench301,p_c=0.5):
    r  = random.uniform(0,1)
    x1,x2 = nasbench.crossover(p1.X["arch"],p2.X["arch"],p_c)
    ind1 = Individual(X={"arch":x1},age=0)
    ind2 = Individual(X={"arch":x2},age=0)
    return ind1,ind2 

def mutate_ind(p,nasbench:Nasbench301,p_m=0.05):
    r = random.uniform(0,1)
    new_arch = nasbench.mutate(p.X["arch"],p_m)
    ind = Individual(X={"arch":new_arch},age=0)
    return ind

def cos_dis(a, b):
    return 1-np.matmul(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

class ENAS(object):
    def __init__(self,nasspace,g2v_model,args):
        self.pop_size = args.pop_size
        self.total_gen = args.total_gen
        self.total_eval = args.total_eval
        self.nasspace = nasspace
        self.args = args
        self.seed = args.seed
        self.p_c = args.p_c
        self.p_m = args.p_m

        self.n_gen = 0
        self.n_eval = 0
        self.best_F = 0.
        self.best_FS = []
        self.test_F = 0.
        self.test_FS = []
        self.pop = []
        self.archive = []
        self.hash_visited = {}
        self.n_feature = 32
        self.code_type = "adj"
        self.g2v_model = g2v_model
        self.n_cluster = 20
        self.K = 10
        self.W = 4
        self.M = 6
        
        self.ranknet=None
        self.gpr = None

        self.parent_diversity = []
        self.off_diversity = []
        
        self.diversitys = []


    def initialize(self):
        while len(self.archive)<self.pop_size:
            arch = Nasbench301.get_cell().random_cell(self.nasspace.nasbench,random_encoding='adj')
            hash_arch  = self.nasspace.get_hash(arch)
            if hash_arch in self.hash_visited:
                continue
            else: self.hash_visited[hash_arch]=1
            F = 100-Nasbench301.get_cell(arch).get_val_loss(self.nasspace.nasbench)
            ind  = Individual(X=arch,age=0,F=F,score=F,code=self.encode_g2v(arch))
            self.archive.append(ind)
            if F>self.best_F:
                self.best_F = F
            self.best_FS.append(self.best_F)
            self.n_eval+=1

        #--- 构建ranknet训练的sample ---
        self.train_surrogate(self.archive)
        self.archive = sorted(self.archive,key=lambda x:x.F,reverse=True)

        #--- 构建archive_pop 和 pop
        self.archive_pop = copy.deepcopy(self.archive)
        self.pop = copy.deepcopy(self.archive_pop)
        

    def has_next(self):
        return self.n_eval<self.total_eval
    
    def encode_g2v(self,arch):
        return Nasbench301.get_cell(arch).encode_g2v(self.nasspace,self.g2v_model)

    
    def solve(self):
        self.initialize()
        diversity = self.pop_diversity(self.pop)
        self.diversitys.append(diversity)
        logging.info("gen:{},diversity:{}".format(self.n_gen,diversity)) 
        while self.has_next():
            self.next()
            diversity = self.pop_diversity(self.pop)
            self.diversitys.append(diversity)
            logging.info("gen:{},diversity:{}".format(self.n_gen,diversity)) 
            logging.info("gen:{},FS:{}".format(self.n_gen,[ind.F for ind in self.archive])) 
        return self.best_FS,self.diversitys
    
    def pop_diversity(self,pop):
        cand_X = [ind.code for ind in pop]
        n_cand = len(cand_X)
        total_dis,k=0.,0
        if len(pop)==1:
            return 0.
        for i in range(n_cand):
            for j in range(i+1,n_cand):
                dis = cos_dis(cand_X[i],cand_X[j])
                total_dis+=dis
                k+=1
        return total_dis/k
    
    def train_surrogate(self,pop):

        ## ranknet
        model_pool = [(ind.code,ind.F) for ind in pop]
        random.shuffle(model_pool)
        samples = build_train_sample(model_pool)
        self.ranknet = RankNet(self.n_feature)
        self.ranknet.fit(*samples)

    
    def predict(self,pop,return_std=False):
        scores = []
        xembedding = [ind.code for ind in pop]
        predicted= np.squeeze(self.ranknet.predict(xembedding).detach().cpu().numpy())
        if return_std:
            for _ in range(5):
                xembedding = [self.encode_g2v(ind.X) for ind in pop]
                score = np.squeeze(self.ranknet.predict(xembedding).detach().cpu().numpy())
                scores.append(score)
            return predicted,np.std(scores,axis=0)
        else:
            return predicted

    def next(self):
        offspring = []
        hash_visited = copy.deepcopy(self.hash_visited)
        logging.info("size of pop:{}".format(len(self.pop)))
        num_mutated = 0
        offspring_size = self.pop_size*self.M
        for ind in self.pop:
            hash_ind = self.nasspace.get_hash(ind.X)
            if hash_ind not in hash_visited:
                hash_visited[hash_ind]=1
        logging.info("size of hash_visited:{}".format(len(hash_visited)))

        patience = 100
        num_mutated =0
        while len(offspring)<self.pop_size*self.M:
            if patience==0:
                break
            p1,p2 = tournament_select(self.pop,n_sample=2),tournament_select(self.pop,n_sample=2)
            p1,p2 = crossover_ind(p1,p2,nasbench=self.nasspace)
            p1 = mutate_ind(p1,nasbench=self.nasspace)
            p2 = mutate_ind(p2,nasbench=self.nasspace)
            hash_p1 = self.nasspace.get_hash(p1.X)
            hash_p2 = self.nasspace.get_hash(p2.X)
            if (hash_p1 in hash_visited) and (hash_p2 in hash_visited):
                patience-=1
            if hash_p1 not in hash_visited:
                offspring.append(p1)
                hash_visited[hash_p1]=1
                patience=100
            if hash_p2 not in hash_visited:
                offspring.append(p2)
                hash_visited[hash_p2]=1
                patience=100
            num_mutated+=1
        logging.info("num_mutated:{}".format(num_mutated))

        if len(offspring)<self.pop_size*self.M:
            logging.info("The number of offspring is insufficient, uniform sample ")
        while len(offspring)<self.pop_size*self.M:
            arch = Nasbench301.get_cell().random_cell(self.nasspace.nasbench,random_encoding='adj')
            hash_arch = self.nasspace.get_hash(arch)
            if hash_arch not in hash_visited:
                offspring.append(Individual(X=arch,age=0))
                hash_visited[hash_arch]=1
        logging.info("offspring is ready.")
        mixed = Population.merge(self.pop,offspring)
        for ind in mixed:
            if ind.code is None:
                ind.code = self.encode_g2v(ind.X)
        scores = self.predict(mixed)
        # trues=[100-Nasbench301.get_cell(ind.X).get_val_loss(nasbench=self.nasspace.nasbench) for ind in mixed]
        # logging.info("the trues is ready.")
        # kd = kendalltau(scores,trues)[0]
        # logging.info("kd:{}".format(kd))
        for i in range(len(mixed)):
            mixed[i].score = scores[i]
        diss = np.full((self.pop_size,self.pop_size*self.M),np.inf)
        n_update = 0
        for i in range(self.pop_size):
            for j in range(self.pop_size*self.M):
                diss[i,j] = cos_dis(self.pop[i].code,offspring[j].code)
        associate_stat = np.full((self.pop_size,),0)
        associate_list = [[] for i in range(self.pop_size)]
        for _ in range(self.pop_size*self.M):
            xs,ys = np.where(diss==np.min(diss))
            x,y = xs[0],ys[0]
            associate_stat[x]+=1
            associate_list[x].append(y)
            diss[:,y]=np.inf
            if associate_stat[x]==self.M:
                diss[x,:]=np.inf
        for i in range(self.pop_size):
            associate = [offspring[j] for j in associate_list[i]]
            associate_scores = [ind.score for ind in associate]
            best_id = np.argmax(associate_scores)
            if self.n_gen%self.W==0:
                self.pop[i] = associate[best_id]
                n_update+=1
            elif self.pop[i].score<associate[best_id].score:
                self.pop[i] = associate[best_id]
                n_update+=1
        logging.info("gen:{} n_update:{}".format(self.n_gen+1,n_update))
        logging.info("scores:{}".format(np.sort(-scores)[:self.pop_size]))
        
        
        if (self.n_gen+1)%self.W==0:
            self.pop = sorted(self.pop,key=lambda x:x.score,reverse=True)
            scores_infill,uncerit_infill = self.predict(self.pop,return_std=True)
            logging.info("scores_infill:{}".format(scores_infill))
            logging.info("uncerit_info:{}".format(uncerit_infill))
            for i in range(len(self.pop)):
                self.pop[i].score = scores_infill[i]
                self.pop[i].uncerit = uncerit_infill[i]
            
            ids_sorted = self.infill(self.pop)
            k=0
            new_candidate = []
            for id in ids_sorted:
                ind = self.pop[id]
            # for ind in self.pop:
                if ind.F is None:
                    ind.F = 100-Nasbench301.get_cell(ind.X).get_val_loss(nasbench=self.nasspace.nasbench)
                    self.archive.append(ind)
                    new_candidate.append(ind)
                    self.hash_visited[self.nasspace.get_hash(ind.X)]=1
                    self.n_eval+=1
                    k+=1
                if k>=self.K:
                    break
            self.train_surrogate(self.archive)

            ##--- 更新下一代种群 ---
            num_resample = len(new_candidate)
            self.archive_pop = sorted(self.archive_pop,key=lambda x:x.F,reverse=True)
            n_absolate = self.pop_size-num_resample
            diss = np.full((num_resample,num_resample),0.)
            for i in range(num_resample):
                for j in range(num_resample):
                    diss[i,j] = cos_dis(self.archive_pop[n_absolate+i].code,new_candidate[j].code)
            n_update = 0
            for i in range(num_resample):
                xs,ys = np.where(diss==np.min(diss))
                x,y = xs[0],ys[0]
                if self.archive_pop[n_absolate+x].F<=new_candidate[y].F:
                    self.archive_pop[n_absolate+x] = new_candidate[y]
                    n_update+=1
                diss[:,y]=np.inf
                diss[x,:]=np.inf
            self.pop = copy.deepcopy(self.archive_pop)
            for ind in self.pop:
                ind.score = ind.F
            
            logging.info("gen:{} archive_pop n_update:{}".format(self.n_gen+1,n_update))

            #--- 显示搜索到的最优个体 ---
            self.archive = sorted(self.archive,key=lambda x:x.F,reverse=True)
            self.best_F = self.archive[0].F
            self.best_FS.extend([self.best_F]*k)
            logging.info("gen:{} n_eval:{} best_F:{}".format(self.n_gen,self.n_eval,self.best_F))

            # ## 添加不确定性
            # diversity = self.pop_diversity(self.archive)
            # self.diversitys.append(diversity)
        
        self.n_gen+=1
    
    def infill(self,pop):
        scores = [ind.score for ind in pop]
        uncerit = [ind.uncerit for ind in pop]
        su = np.array([scores,uncerit])
        F,rank = NonDominatedSorting(su)
        selected_id = []
        logging.info("nondo_sort F:{}".format(F))
        for i in range(len(F)):
            selected_id.extend(F[i])
        return selected_id

 

