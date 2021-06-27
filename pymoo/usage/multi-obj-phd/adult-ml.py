import string

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.crossover import Crossover
from pymoo.model.duplicate import ElementwiseDuplicateElimination
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pyitlib import discrete_random_variable as drv
import math
import random
from classifier import binaryClassifier


class MyProblem(Problem):

    def __init__(self):
        c = binaryClassifier()
        self.classifier = c
        self.stat = []
        self.length_gene = 100
        self.n_genes = 8
        self.bounds = ((17,90),(1,16),(0,1),(0,4),(0,1),(0,99999),(0,4356),(1,99)) # variables' original bounds (int)

        super().__init__(n_var=self.n_genes, 
                         n_obj=2, 
                         n_constr=1, 
                         elementwise_evaluation=True)

    def _evaluate(self, chromosome, out, *args, **kwargs):

        y=[]
        #predict
        #len(chromosome) = 30 (observations)
        for i in range(self.length_gene):
            y.append(self.classifier.predict([chromosome[i]]))
        
        #compute mutual array using y
        M=[]
        ch = np.array(chromosome)
        
        for i in range(self.num_genes):
            M.append(drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True))
   
        #dictionary Variables-MI
        Var = ["age","education.num","marital.status","race","sex","capital.gain","capital.loss","hours.per.week"]

        d = {"".join(Var[0]):M[0]}
        #print(d)

        for i in range(len(Var)):
            d["".join(Var[i])] = M[i]
        #print(d)

        sort_orders = sorted(d.items(), key=lambda x: x[1], reverse=True)
        #print(sort_orders)

        best = []

        den = 0
        num = 0
        res = 0
        summary = 0
        summary = sum(M)
        threshold = summary * 0.5
        
        M.sort(reverse=True)
        #print(M)

        #den = 0
        while num < threshold:
            num=num+M[den]
            den=den+1

        for i in range(den):
            best.append(sort_orders[i])
        self.stat.append([res,den,best])

        f1 = num
        f2 = -den
        g1 = den<0

        out["F"] = [f1, f2]
        out["G"] = [g1]

class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):

        pop = []

        for i in range (n_samples):
            X = np.full((problem.length_gene, problem.n_genes), 0, int)

            for i in range(len(X)):
                for j in  range(len(X[i])):
                    X[i][j] = random.randint(problem.bounds[j][0],problem.bounds[j][1])
            pop.append(X)

        print("population:", pop)
        return pop[0]

# class MyCrossover(Crossover):
#     def __init__(self):
#         super().__init__(2, 2)

#     def _do(self, problem, X, parents, **kwargs):

#         # a,b = example_parents(2,8)

#         # print("One Point Crossover")
#         # off = crossover(get_crossover("bin_one_point"), a, b)
#         # show((off[:n_matings] != a[0]))

#         print("crossover:", Y)
#         return Y


algorithm = NSGA2(pop_size=20,
                  sampling=MySampling(),
                  #crossover=MyCrossover(),
                  #mutation=MyMutation(),
                  #eliminate_duplicates=MyDuplicateElimination()
                  )

res = minimize(MyProblem(),
               algorithm,
               seed=1,
               verbose=True)

Scatter().add(res.F).show()
print(res.X[np.argsort(res.F[:, 0])])