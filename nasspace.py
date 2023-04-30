import os
import copy
import numpy as np
import pickle
import random

from nascell.cell_301 import Cell301
from nascell.cell_201 import Cell201
from nascell_101.cell_101 import Cell101

from nasbench import api as nb101
from nas_201_api import NASBench201API as nb201
import nasbench301.api as nb301
from operations import OPERATIONS_201, OPERATIONS_101

default_data_folder = '/data/Fanliang/data/'

class Nasbench:

    def get_cell(self, arch=None):
        return None

    def query_arch(self, 
                   arch=None, 
                   train=True, 
                   predictor_encoding=None, 
                   cutoff=0,
                   random_encoding='adj',
                   deterministic=True,
                   epochs=0,
                   random_hash=False,
                   max_edges=None,
                   max_nodes=None):

        arch_dict = {}
        arch_dict['epochs'] = epochs

        if arch is None:

            arch = self.get_cell().random_cell(self.nasbench,
                                               random_encoding=random_encoding, 
                                               max_edges=max_edges, 
                                               max_nodes=max_nodes,
                                               cutoff=cutoff,
                                               index_hash=self.index_hash)
        arch_dict['spec'] = arch

        if predictor_encoding:
            arch_dict['encoding'] = self.get_cell(arch).encode(predictor_encoding=predictor_encoding,
                                                                 nasbench=self.nasbench,
                                                                 deterministic=deterministic,
                                                                 cutoff=cutoff)

        if train:
            arch_dict['val_loss'] = self.get_cell(arch).get_val_loss(self.nasbench, 
                                                                       deterministic=deterministic,
                                                                       dataset=self.dataset)
            arch_dict['test_loss'] = self.get_cell(arch).get_test_loss(self.nasbench,
                                                                         dataset=self.dataset)
            arch_dict['num_params'] = self.get_cell(arch).get_num_params(self.nasbench)
            arch_dict['val_per_param'] = (arch_dict['val_loss'] - 4.8) * (arch_dict['num_params'] ** 0.5) / 100

        return arch_dict

    def mutate_arch(self, 
                    arch, 
                    mutation_rate=1.0, 
                    mutate_encoding='adj',
                    cutoff=0):

        return self.get_cell(arch).mutate(self.nasbench,
                                            mutation_rate=mutation_rate,
                                            mutate_encoding=mutate_encoding,
                                            index_hash=self.index_hash,
                                            cutoff=cutoff)

    def generate_random_dataset(self,
                                num=10, 
                                train=True,
                                predictor_encoding=None, 
                                random_encoding='adj',
                                deterministic_loss=True,
                                patience_factor=5,
                                allow_isomorphisms=False,
                                cutoff=0,
                                max_edges=None,
                                max_nodes=None):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break

            arch_dict = self.query_arch(train=train,
                                        predictor_encoding=predictor_encoding,
                                        random_encoding=random_encoding,
                                        deterministic=deterministic_loss,
                                        cutoff=cutoff,
                                        max_edges=max_edges,
                                        max_nodes=max_nodes)

            h = self.get_hash(arch_dict['spec'])

            if allow_isomorphisms or h not in dic:
                dic[h] = 1
                data.append(arch_dict)
        return data


    def get_candidates(self, 
                       data, 
                       num=100,
                       acq_opt_type='mutation',
                       predictor_encoding=None,
                       mutate_encoding='adj',
                       loss='val_loss',
                       allow_isomorphisms=False, 
                       patience_factor=5, 
                       deterministic_loss=True,
                       num_arches_to_mutate=1,
                       max_mutation_rate=1,
                       train=False,
                       cutoff=0):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """

        candidates = []
        # set up hash map
        dic = {}
        for d in data:
            arch = d['spec']
            h = self.get_hash(arch)
            dic[h] = 1

        if acq_opt_type not in ['mutation', 'mutation_random', 'random']:
            print('{} is not yet implemented as an acquisition type'.format(acq_opt_type))
            raise NotImplementedError()

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest loss
            best_arches = [arch['spec'] for arch in sorted(data, key=lambda i:i[loss])[:num_arches_to_mutate * patience_factor]]

            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime

            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(int(num / num_arches_to_mutate / max_mutation_rate)):
                    for rate in range(1, max_mutation_rate + 1):
                        mutated = self.mutate_arch(arch, 
                                                   mutation_rate=rate, 
                                                   mutate_encoding=mutate_encoding)
                        arch_dict = self.query_arch(mutated, 
                                                    train=train,
                                                    predictor_encoding=predictor_encoding,
                                                    deterministic=deterministic_loss,
                                                    cutoff=cutoff)
                        h = self.get_hash(mutated)

                        if allow_isomorphisms or h not in dic:
                            dic[h] = 1    
                            candidates.append(arch_dict)

        if acq_opt_type in ['random', 'mutation_random']:
            # add randomly sampled architectures to the set of candidates
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break

                arch_dict = self.query_arch(train=train, 
                                            predictor_encoding=predictor_encoding,
                                            cutoff=cutoff)
                h = self.get_hash(arch_dict['spec'])

                if allow_isomorphisms or h not in dic:
                    dic[h] = 1
                    candidates.append(arch_dict)

        return candidates

    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            dic[self.get_hash(d['spec'])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_hash(candidate['spec']) not in dic:
                dic[self.get_hash(candidate['spec'])] = 1
                unduplicated.append(candidate)
        return unduplicated

    def train_test_split(self, data, train_size, 
                                     shuffle=True, 
                                     rm_duplicates=True):
        if shuffle:
            np.random.shuffle(data)
        traindata = data[:train_size]
        testdata = data[train_size:]

        if rm_duplicates:
            self.remove_duplicates(testdata, traindata)
        return traindata, testdata


    def encode_data(self, dicts):
        # input: list of arch dictionary objects
        # output: xtrain (in binary path encoding), ytrain (val loss)

        data = []
        for dic in dicts:
            arch = dic['spec']
            encoding = Arch(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))
        return data

    def get_arch_list(self,
                      aux_file_path,
                      distance=None,
                      iteridx=0,
                      num_top_arches=5,
                      max_edits=20,
                      num_repeats=5,
                      random_encoding='adj',
                      verbose=0):
        # Method used for gp_bayesopt

        # load the list of architectures chosen by bayesopt so far
        base_arch_list = pickle.load(open(aux_file_path, 'rb'))
        top_arches = [archtuple[0] for archtuple in base_arch_list[:num_top_arches]]
        if verbose:
            top_5_loss = [archtuple[1][0] for archtuple in base_arch_list[:min(5, len(base_arch_list))]]
            print('top 5 val losses {}'.format(top_5_loss))

        # perturb the best k architectures    
        dic = {}
        for archtuple in base_arch_list:
            path_indices = self.get_cell(archtuple[0]).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    perturbation = self.get_cell(arch).mutate(self.nasbench, edits)
                    path_indices = self.get_cell(perturbation).get_path_indices()
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append(perturbation)

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                arch = self.get_cell().random_cell(self.nasbench, random_encoding=random_encoding)
                path_indices = self.get_cell(arch).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(arch)

        return new_arch_list

    # Method used for gp_bayesopt for nasbench
    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        matrix = np.zeros([len(arches_1), len(arches_2)])
        for i, arch_1 in enumerate(arches_1):
            for j, arch_2 in enumerate(arches_2):
                matrix[i][j] = cls.get_cell(arch_1).distance(cls.get_cell(arch_2), dist_type=distance)
        return matrix

class Nasbench101(Nasbench):

    def __init__(self,
               data_folder=default_data_folder,
               index_hash_folder='./data/',
               mf=True):
        self.mf = mf
        self.dataset = 'cifar10'

        """
        For NAS encodings experiments, some of the path-based encodings currently require a
        hash map from path indices to cell architectuers. We have created a pickle file which
        contains the hash map, located at 
        https://drive.google.com/file/d/1yMRFxT6u3ZyfiWUPhtQ_B9FbuGN3X-Nf/view?usp=sharing
        """
        self.index_hash = None
        index_hash_path = os.path.expanduser(index_hash_folder + 'index_hash.pkl')
        if os.path.isfile(index_hash_path):
            self.index_hash = pickle.load(open(index_hash_path, 'rb'))

        if not self.mf:
            self.nasbench = nb101.NASBench(data_folder + 'nasbench_only108.tfrecord')
        else:
            self.nasbench = nb101.NASBench(data_folder + 'nasbench_full.tfrecord')
        self.NUM_VERTICES = 7
        self.OP_SPOTS = self.NUM_VERTICES - 2
        self.MAX_EDGES = 9
    
    def get_arch_iter(self):
        for unique_hash in self.nasbench.hash_iterator():
            matrix = self.nasbench.fixed_statistics[unique_hash]['module_adjacency']
            ops = self.nasbench.fixed_statistics[unique_hash]['module_operations']
            arch = self.get_cell().convert_to_cell({"matrix":matrix,"ops":ops})
            yield unique_hash,arch
    
    def __len__(self):
        return 423624

    @classmethod
    def get_cell(cls, arch=None):
        if not arch:
            return Cell101
        else:
            return Cell101(**arch)

    def get_type(self):
        return 'nasbench_101'

    def get_nbhd(self, arch, mutate_encoding='adj'):
        return Cell101(**arch).get_neighborhood(self.nasbench, 
                                                mutate_encoding=mutate_encoding,
                                                index_hash=self.index_hash)

    def get_hash2(self, arch):
        # return a unique hash of the architecture+fidelity
        return Cell101(**arch).get_path_indices()
    
    def get_hash(self,arch):
        path_indices =  Cell101(**arch).encode_paths_seq_aware(length=120)
        path_indices = ''.join(map(str, path_indices.tolist()))
        return path_indices

    def crossover(self,X1,X2,index_hash=None,crossover_rate=0.5):
        ## 5、path index_hash crossover
        # p=0
        # while p<100:
        #     p+=1
        #     path_indices_1 = list(Cell101(X1['matrix'],X1["ops"]).get_path_indices())
        #     n_path_1 = len(path_indices_1)
        #     path_indices_2 = list(Cell101(X2["matrix"],X2["ops"]).get_path_indices())
        #     n_path_2 = len(path_indices_2)
        #     path_indices = path_indices_1+path_indices_2
        #     random.shuffle(path_indices)
        #     print("path_indices",path_indices)
        #     new_path_indice_1 = tuple(np.unique(path_indices[:n_path_1]))
        #     new_path_indice_2 = tuple(np.unique(path_indices[n_path_1:]))
        #     if (new_path_indice_1 in index_hash) and (new_path_indice_2 in index_hash):
        #         spec1 = index_hash[new_path_indice_1]
        #         matrix1,ops1 = spec1['matrix'],spec1['ops']
        #         model_spec1 = api.ModelSpec(matrix=matrix1, ops=ops1)

        #         spec2 = index_hash[new_path_indice_2]
        #         matrix2,ops2 = spec2['matrix'],spec2['ops']
        #         model_spec2 = api.ModelSpec(matrix=matrix2, ops=ops2)

        #         if self.nasbench.is_valid(model_spec1) and self.nasbench.is_valid(model_spec2):
        #             return {
        #                     'matrix': matrix1,
        #                     'ops': ops1
        #                     },{
        #                     'matrix': matrix2,
        #                     'ops': ops2 
        #                     }
        # print("crossover is not valid")
        # return {
        #         'matrix': copy.deepcopy(X1['matrix']),
        #         'ops': copy.deepcopy(X1['ops'])
        #         },{
        #         'matrix': copy.deepcopy(X2['matrix']),
        #         'ops': copy.deepcopy(X2['ops'])
        #         }

        # 4、path crossover
        p = 0
        matrix_A = copy.deepcopy(X1['matrix'])
        ops_A = copy.deepcopy(X1['ops'])
        matrix_B = copy.deepcopy(X2['matrix'])
        ops_B = copy.deepcopy(X2['ops'])
        paths = Cell101(matrix_A,ops_A).get_path_position()
        n_path_A = len(paths)
        # print("pathA:{}".format(paths))
        paths2 = Cell101(matrix_B,ops_B).get_path_position()
        # print("pathA:{}".format(paths2))
        paths.extend(paths2)
        n_path_B = len(paths)-n_path_A
        while p<100:
            p+=1
            new_matrix_A = np.full(matrix_A.shape,0)
            new_matrix_B = np.full(matrix_B.shape,0)
            # selected_id_A =  random.sample(range(n_path_A+n_path_B),k=n_path_A)
            # selected_id_B = random.sample(range(n_path_A+n_path_B),k=n_path_B)
            selected_ids = list(range(n_path_A+n_path_B))
            random.shuffle(selected_ids)
            selected_id_A = selected_ids[:n_path_A]
            selected_id_B = selected_ids[n_path_A:]
            new_path_A  = [paths[id] for id in selected_id_A]
            # print("new_path_A:{}".format(new_path_A))
            new_path_B = [paths[id] for id in selected_id_B]
            # print("new_path_B:{}".format(new_path_B))
            for path in new_path_A:
                for (x,y) in path:
                    new_matrix_A[x,y] = 1
            for path in new_path_B:
                for (x,y) in path:
                    new_matrix_B[x,y]=1
            new_ops_A,new_ops_B = [],[]
            #  得到o1的ops
            count_visite_A = np.full((len(ops_A),),0)
            count_visite_B = np.full((len(ops_B),),0)
            for id in selected_id_A:
                if id <n_path_A:
                    #选中了P_A
                    for x,y in paths[id]:
                        count_visite_A[x]+=1
                else:
                    #选中了B
                    for x,y in paths[id]:
                        count_visite_B[x]+=1
            zerios_id = np.logical_and(count_visite_A==0 , count_visite_B==0)
            count_visite_A[zerios_id]=1
            count_visite_B[zerios_id]=1
            option_ops_A = np.random.uniform(0,1)< count_visite_A/(count_visite_A+count_visite_B)
            for i in range(len(option_ops_A)):
                if option_ops_A[i]:
                    new_ops_A.append(ops_A[i])
                else:
                    new_ops_A.append(ops_B[i])
            
            #  得到o2的ops
            count_visite_A = np.full((len(ops_A),),0)
            count_visite_B = np.full((len(ops_B),),0)
            for id in selected_id_B:
                if id <n_path_A:
                    #选中了P_A
                    for x,y in paths[id]:
                        count_visite_A[x]+=1
                else:
                    #选中了B
                    for x,y in paths[id]:
                        count_visite_B[x]+=1
            zerios_id = np.logical_and(count_visite_A==0 , count_visite_B==0)
            count_visite_A[zerios_id]=1
            count_visite_B[zerios_id]=1
            option_ops_A = np.random.uniform(0,1)< count_visite_A/(count_visite_A+count_visite_B)
            for i in range(len(option_ops_A)):
                if option_ops_A[i]:
                    new_ops_B.append(ops_A[i])
                else:
                    new_ops_B.append(ops_B[i])
            new_spec_A = nb101.ModelSpec(new_matrix_A, new_ops_A)
            new_spec_B = nb101.ModelSpec(new_matrix_B, new_ops_B)
            if self.nasbench.is_valid(new_spec_A) and self.nasbench.is_valid(new_spec_B):
                    # logging.info("The crossover operation has already taken place")
                return {
                    'matrix': new_matrix_A,
                    'ops': new_ops_A
                },{
                'matrix': new_matrix_B,
                'ops': new_ops_B 
                }
        return {
                'matrix': matrix_A,
                'ops': ops_A
                },{
                'matrix': matrix_B,
                'ops': ops_B 
                }

        # p=0
        # matrix_A = copy.deepcopy(X1['matrix'])
        # ops_A = copy.deepcopy(X1['ops'])
        # matrix_B = copy.deepcopy(X2['matrix'])
        # ops_B = copy.deepcopy(X2['ops'])
        # while p<100:
        #     p+=1
        #     new_matrix_A = copy.deepcopy(matrix_A)
        #     new_ops_A = copy.deepcopy(ops_A)
        #     new_matrix_B = copy.deepcopy(matrix_B)
        #     new_ops_B = copy.deepcopy(ops_B)
            
        #     # 3.
        #     cross_points = np.random.uniform(0,1,size=len(ops_A))<crossover_rate
        #     new_matrix_A[:,cross_points],new_matrix_B[:,cross_points]=\
        #     new_matrix_B[:,cross_points].copy(),new_matrix_A[:,cross_points].copy()
        #     for k in range(len(cross_points)):
        #         if cross_points[k]:
        #             new_ops_A[k],new_ops_B[k] = new_ops_B[k],new_ops_A[k]
        #     # 2.
        #     # o_c = np.random.uniform(0,1)
        #     # m_c = np.random.uniform(0,1)
        #     # if o_c<0.5:
        #     #     cross_ops_points = np.random.uniform(0,1,size=len(ops_A))<crossover_rate
        #     #     for k in range(len(cross_ops_points)):
        #     #         if cross_ops_points[k]:
        #     #             new_ops_A[k],new_ops_B[k] = new_ops_B[k],new_ops_A[k]
        #     # if o_c>=0.5 or m_c<0.5:
        #     #     cross_mat_points = np.random.uniform(0,1,size=len(ops_A))<crossover_rate
        #     #     new_matrix_A[:,cross_mat_points],new_matrix_B[:,cross_mat_points]=\
        #     #     new_matrix_B[:,cross_mat_points].copy(),new_matrix_A[:,cross_mat_points].copy()
        #     # 1.
        #     # cross_mat_points = np.random.uniform(0,1,size=matrix_A.shape)<crossover_rate
        #     # cross_ops_points = np.random.uniform(0,1,size=len(ops_A))<crossover_rate
        #     # new_matrix_A = np.select([cross_mat_points,~cross_mat_points],[matrix_A,matrix_B])
        #     # new_matrix_B = np.select([~cross_mat_points,cross_mat_points],[matrix_A,matrix_B])
        #     # new_ops_A,new_ops_B = [],[]
        #     # for k in range(cross_ops_points.shape[0]):
        #     #     if cross_ops_points[k]:
        #     #         new_ops_A.append(ops_B[k])
        #     #         new_ops_B.append(ops_A[k])
        #     #     else:
        #     #         new_ops_A.append(ops_A[k])
        #     #         new_ops_B.append(ops_B[k])
        #     new_spec_A = api.ModelSpec(new_matrix_A, new_ops_A)
        #     new_spec_B = api.ModelSpec(new_matrix_B, new_ops_B)
        #     if self.nasbench.is_valid(new_spec_A) and self.nasbench.is_valid(new_spec_B):
        #         # logging.info("The crossover operation has already taken place")
        #         return {
        #             'matrix': new_matrix_A,
        #             'ops': new_ops_A
        #         },{
        #         'matrix': new_matrix_B,
        #         'ops': new_ops_B 
        #         }
        # return {
        #         'matrix': matrix_A,
        #         'ops': ops_A
        #         },{
        #         'matrix': matrix_B,
        #         'ops': ops_B 
        #         }
    
    def mutate(self,X1,p_m=0.05):
        matrix = copy.deepcopy(X1['matrix'])
        ops = copy.deepcopy(X1['ops'])
        p=0
        while p<200:
            p+=1
            mutate_mat_points = np.random.uniform(0,1,size=matrix.shape)<p_m
            mutate_op_points = np.random.uniform(0,1,size=len(ops))<p_m
            new_matrix = np.select([mutate_mat_points,~mutate_mat_points],[matrix,1-matrix])
            new_matrix = np.triu(new_matrix,k=1)
            new_ops = copy.deepcopy(ops)
            for point in range(len(mutate_op_points)):
                if mutate_op_points[point]:
                    available = [o for o in OPERATIONS_101 if o != ops[point]]
                    new_ops[point] = np.random.choice(available)
            new_spec = nb101.ModelSpec(new_matrix, new_ops)
            if self.nasbench.is_valid(new_spec):
                # logging.info("The mutation operation has already taken place")
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                    }
        return {
                'matrix': matrix,
                'ops': ops
                }


class Nasbench201(Nasbench):

    def __init__(self,
                 dataset='cifar10',
                 data_folder=default_data_folder,
                 version='1_0'):
        self.search_space = 'nasbench_201'
        self.dataset = dataset
        self.index_hash = None
        
        if version == '1_0':
            self.nasbench = nb201(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_0-e61699.pth'),verbose=False)
        elif version == '1_1':
            self.nasbench = nb201(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_1-096897.pth'),verbose=False)
        
    def query_index_by_ops(self,ops):
        return self.nasbench.query_index_by_arch(Cell201.get_string_from_ops(ops))

    def get_type(self):
        return 'nasbench_201'

    @classmethod
    def get_cell(cls, arch=None):
        if not arch:
            return Cell201
        elif isinstance(arch,dict):
            return Cell201(**arch)
        else:
            return Cell201(arch)

    def get_nbhd(self, arch, mutate_encoding='adj'):
        return Cell201(**arch).get_neighborhood(self.nasbench, 
                                                mutate_encoding=mutate_encoding)

    def get_hash(self, arch):
        # return a unique hash of the architecture+fidelity
        return Cell201(**arch).get_string()
    
    def crossover(self,X1,X2,p_c):
        #2、one-point crossover
        # ops1 = copy.deepcopy(self.get_cell(X1).get_op_list())
        # ops2 = copy.deepcopy(self.get_cell(X2).get_op_list())
        # len_code = len(ops1)
        # point = np.random.randint(1,len_code)
        # ops1[point:],ops2[point:]  = ops2[point:],ops1[point:]
        # o1 = self.get_cell().get_string_from_ops(ops1)
        # o2 = self.get_cell().get_string_from_ops(ops2)
        # return {"string":o1},{"string":o2}

        #1、uniform crossover
        ops1 = copy.deepcopy(self.get_cell(X1).get_op_list())
        ops2 = copy.deepcopy(self.get_cell(X2).get_op_list())
        len_code = len(ops1)
        crossover_points = np.random.uniform(0,1,len_code)<p_c
        for point in range(len_code):
            if crossover_points[point]:
                ops1[point],ops2[point] = ops2[point],ops1[point]
        o1 = self.get_cell().get_string_from_ops(ops1)
        o2 = self.get_cell().get_string_from_ops(ops2)
        return o1,o2
    
    def mutate(self,X,p_m):
        ops = copy.deepcopy(self.get_cell(X).get_op_list())
        len_ops = len(ops)
        ##  1、uniform
        mutate_points = np.random.uniform(0.,1.,len_ops)<p_m
        for point in range(len_ops):
            if mutate_points[point]:
                available = [o for o in OPERATIONS_201 if o != ops[point]]
                ops[point] = np.random.choice(available)

        # 2、single
        # id_op = random.randint(0,len_ops-1)
        # available = [o for o in OPS if o!=ops[id_op]]
        # ops[id_op] = random.choice(available)
        
        # 3、uniform muate_rate = 1/len
        # mutate_points = np.random.uniform(0.,1.,len_ops)<1/len_ops
        # for point in range(len_ops):
        #     if mutate_points[point]:
        #         available = [o for o in OPS if o != ops[point]]
        #         ops[point] = np.random.choice(available)
        
        o1 = self.get_cell().get_string_from_ops(ops) 
        return o1

class Nasbench301(Nasbench):

    def __init__(self):
        self.OPS = ['max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5'
            ]
        self.NUM_VERTICES = 4
        self.INPUT_1 = 'c_k-2'
        self.INPUT_2 = 'c_k-1'
        self.OUTPUT = 'c_k'

        self.dataset = 'cifar10'
        self.search_space = 'nasbench_301'
        ensemble_dir_performance = os.path.expanduser('nb_models/xgb_v1.0')
        performance_model = nb301.load_ensemble(ensemble_dir_performance)
        ensemble_dir_runtime = os.path.expanduser('nb_models/lgb_runtime_v1.0')
        runtime_model = nb301.load_ensemble(ensemble_dir_runtime)
        self.nasbench = [performance_model, runtime_model] 
        self.index_hash = None

    def get_type(self):
        return 'nasbench_301'

    @classmethod
    def get_cell(cls, arch=None):
        if not arch:
            return Cell301
        else:
            return Cell301(**arch)

    def get_nbhd(self, arch, mutate_encoding='adj'):
        return Cell301(**arch).get_neighborhood(self.nasbench, 
                                                mutate_encoding=mutate_encoding)

    def get_hash(self, arch):
        # return a unique hash of the architecture+fidelity
        return str(Cell301(**arch).serialize())
    
    def crossover(self,arch1,arch2,cross_rate):
        x1,x2 = np.array(arch1).copy().reshape(-1,2),np.array(arch2).copy().reshape(-1,2)
        n = x1.shape[0]
        crossover_points = np.random.uniform(0.,1.,(n,))<cross_rate
        for point in range(len(crossover_points)):
            if not crossover_points[point]:
                continue
            x1[point],x2[point] = x2[point].copy(),x1[point].copy()
            if point%2==0:
                x1_in,x2_in = x2[point][0],x1[point][0]
                if x1[point][0] == x1[point+1][0]:
                    x1[point][0] = x1_in
                if x2[point][0] == x2[point+1][0]:
                    x2[point][0] = x2_in
            else:
                x1_in,x2_in = x2[point][0],x1[point][0]
                if x1[point][0] == x1[point-1][0]:
                    x1[point][0] = x1_in
                if x2[point][0] == x2[point-1][0]:
                    x2[point][0] = x2_in
        x1 = x1.tolist()
        x2 = x2.tolist()
        # x1 = (x1[:n//2],x1[n//2:])
        # x2 = (x2[:n//2],x2[n//2:])
        # same
        x1 = (x1[:n//2],copy.deepcopy(x1[:n//2]))
        x2 = (x2[:n//2],copy.deepcopy(x2[:n//2]))
        return x1,x2
    
    def mutate(self,arch,mutate_rate):
        x = np.array(arch).copy().reshape(-1,2)
        n = x.shape[0]*2
        mutate_points = np.random.uniform(0.,1.,(n,))<mutate_rate
        for i in range(len(mutate_points)):
            if not mutate_points[i]:
                continue
            node = i//2
            op = i%2
            if op==1:
                x[node][op] = np.random.choice(len(self.OPS))
            else:
                inputs = node//2 if node <8 else node//2-4
                op_in = np.random.choice(2+inputs)
                if node%2==0:
                    while x[node+1][0]==op_in:
                        op_in = np.random.choice(2+inputs)
                if node%2==1:
                    while x[node-1][0]==op_in:
                        op_in = np.random.choice(2+inputs)
                x[node][op] = op_in
        x = x.tolist()
        # x = (x[:n//4],x[n//4:])
        # same
        x = (x[:n//4],copy.deepcopy(x[:n//4]))
        return x
