
class Individual(object):
    def __init__(self,id=None,X=None,age=None,code=None,F=None,rank=None,score=None,uncerit=None,id_cluster=None) -> None:
        self.id = id
        self.X = X
        self.code=code
        self.F = F
        self.id_cluster = id_cluster
        self.score = score
        self.rank = rank
        self.age=age
        self.uncerit = uncerit
        self.attr = set(self.__dict__.keys())
    
    def set(self,key,value):
        if key in self.attr:
            self.__dict__[key] = value
        return self