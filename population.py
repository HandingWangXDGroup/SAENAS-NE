import numpy as np


from individual import Individual
class Population(np.ndarray):
    def __new__(cls,n_individuals=0):
        obj = super(Population,cls).__new__(cls,n_individuals,dtype=cls).view(cls)
        for i in range(n_individuals):
            obj[i]  = Individual()
        return obj
    
    def set(self,ids,archs,FS,scores,**kwargs):
        if self.size == 0:
            return 
        assert len(self) == len(archs)
        for i in range(len(self)):
            self[i].set("id",ids[i])
            self[i].set("X",archs[i])
            self[i].set("F",FS[i])
            self[i].set("score",scores[i])
        return self
    
    def getFS(self):
        FS = np.full((len(self),),0.)
        for i in range(len(self)):
            FS[i] = self[i].F
        return FS
    
    def getArchs(self):
        archs = []
        for i in range(len(self)):
            archs.append(self[i].X)
        return archs
    
    def has_arch(self,arch):
        return arch in self.getArchs()
    
    def clear(self):
        clear_ids = []
        N = len(self)
        for i in range(N):
            for j in range(i+1,N):
                if i in clear_ids or j in clear_ids:
                    continue
                if self[i].X == self[j].X:
                    clear_ids.append(j)
        return np.delete(self,clear_ids,axis=0)

    @classmethod
    def merge(cls,a,b):
        if len(a)==0:
            return b
        elif len(b)==0:
            return a
        else:
            obj = np.concatenate([a,b]).view(Population)
            return obj