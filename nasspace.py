import os
import copy
import numpy as np
import pickle

from nasbench_301.cell_301 import Cell301
import nasbench301.api as nb301

OPS = ['max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5'
       ]
NUM_VERTICES = 4
INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'
OUTPUT = 'c_k'

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

class Nasbench301(Nasbench):

    def __init__(self):
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
                x[node][op] = np.random.choice(len(OPS))
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