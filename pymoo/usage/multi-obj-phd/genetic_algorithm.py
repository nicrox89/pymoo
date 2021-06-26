import sys
from leap_ec.algorithm import generational_ea
from leap_ec import representation, ops

from classifier import binaryClassifier

from leap_ec.segmented_rep import initializers, decoders
from problems import fitness
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.segmented_rep.initializers import create_segmented_sequence
from leap_ec.segmented_rep.ops import apply_mutation
from leap_ec import decoder 
from leap_ec import probe
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.individual import Individual
from leap_ec import util
from leap_ec.context import context
from toolz import pipe
from leap_ec.int_rep.ops import mutate_randint, individual_mutate_randint
import random
from pyitlib import discrete_random_variable as drv
import numpy as np

from leap_ec.problem import FunctionProblem

classifier = binaryClassifier()
p = fitness(classifier)

result = []

#number of individuals(matrixs)
pop_size = 30
#number of instances for each gene(variable) = number of records(observations) of the matrix
gene_size = 300
#number of eatures
num_genes = 8

#"age","education.num","marital.status","race","sex","capital.gain","capital.loss","hours.per.week"
features = ["age","education.num","marital.status","race","sex","capital.gain","capital.loss","hours.per.week"]

#bounds = ((0, 1), (0, 1), (0, 1), (0, 1)) # 4 variables normalized between 0 and 1
bounds = ((17,90),(1,16),(0,1),(0,4),(0,1),(0,99999),(0,4356),(1,99)) # variables' original bounds (int)


# #THE MATRIX
# seqs = [] # Save sequences for next step
# for i in range(pop_size):
#     seq = create_segmented_sequence(gene_size, create_int_vector(bounds)) # a sample - check float
#     seqs.append(seq)

# p.f(seq)

# ea = generational_ea(generations=10, pop_size=pop_size,

#                      # Solve a MaxOnes Boolean optimization problem
#                      problem=FunctionProblem(p.f, True),

#                      representation=representation.Representation(
#                          # Genotype and phenotype are the same for this task
#                          decoder=decoder.IdentityDecoder(),
#                          # Initial genomes are random real sequences
#                          #initialize=initializers.create_segmented_sequence(gene_size, create_real_vector(bounds))
#                          # Initial genomes are random discrete sequences
#                          initialize=initializers.create_segmented_sequence(gene_size, create_int_vector(bounds))

#                      ),

#                      # The operator pipeline
#                      pipeline=[ops.tournament_selection,
#                                # Select parents via tournament_selection selection
#                                ops.clone,  # Copy them (just to be safe)
#                                #probe.print_individual(prefix='before mutation: '),
#                                # Basic mutation: defaults to a 1/L mutation rate
#                                # segmented vectors mutation
#                                # apply_mutation,
#                                mutate_bitflip,
#                                # discrete vectors mutation
#                                # mutate_randint,
#                                #probe.print_individual(prefix='after mutation: '),
#                                # Crossover with a 40% chance of swapping each gene
#                                ops.uniform_crossover(p_swap=0.4),
#                                ops.evaluate,  # Evaluate fitness
#                                # Collect offspring into a new population
#                                ops.pool(size=pop_size)
#                                #yield (generation_counter.generation(), bsf)
#                                ])


# print('Generation, Best_Individual')
# for i, best in ea:
#     print(f"{i}, {best}")


def init(length, seq_initializer):
    def create():
        return initializers.create_segmented_sequence(gene_size, create_int_vector(bounds))
    return create

#create initial rand population of pop_size individuals

parents = Individual.create_population(n=pop_size,
                                       initialize=init(gene_size, create_int_vector(bounds)),
                                       decoder=decoder.IdentityDecoder(),
                                       problem=FunctionProblem(p.f, True))


# Evaluate initial population = calculate Fitness Function for each infividual in the initial population
parents = Individual.evaluate_population(parents)

# print initial, random population + Fitness Function for each individual
# ****
#util.print_population(parents, generation=0)

# generation_counter is an optional convenience for generation tracking
generation_counter = util.inc_generation(context=context)

#results = []
while generation_counter.generation() < 100:
    p.setStat()
    #sequence of functions, the result of the first one will be the parameter of the next one, and so on
    offspring = pipe(parents,
                     #probe.print_individual(prefix='before tournament: '),
                     ops.tournament_selection,
                     #probe.print_individual(prefix='after tournament: \n'),
                     ops.clone,
                     #mutate_bitflip,
                     #probe.print_individual(prefix='before mutation: '),
                     #individual_mutate_randint,
                     #probe.print_individual(prefix='after mutation: '),
                     #probe.print_individual(prefix='before crossover: \n'),
                     ops.uniform_crossover(p_swap=0.2),
                     #probe.print_individual(prefix='after crossover: \n\n\n'),
                     ops.evaluate,
                     ops.pool(size=len(parents)))  # accumulate offspring

    parents = offspring

    #print(probe.best_of_gen(parents))

    generation_counter()  # increment to the next generation

    #util.print_population(parents, context['leap']['generation'])
    
    count=0
    parents_pairs={}
    #parents_pairs collect position individual and fitness function (for each individual in the current population (in the current generation))
    for i in range(len(parents)):
        parents_pairs[i] = parents[i].fitness
    #sort individuals in the current population in an ascending order (by Fitness Function)
    import operator
    sorted_d = sorted(parents_pairs.items(), key=operator.itemgetter(1))

    #for key, value in sorted_d:
        #print("generation", context['leap']['generation'])
        #count=count+1
        #print("individual", count)
        #print(parents[key].genome)
        #print(parents[key].fitness)
        #print(p.getStat()[key])
    #for individual in parents:
    #    print("generation", context['leap']['generation'])
    #    print(p.getStat()[count])
    #    count=count+1
    #    print("individual", count)
        #print(individual.genome)
    #    print(individual.fitness)
    print("generation", context['leap']['generation'])

    #print worse and best (FF) individual in the population for each generation (showing FF + num meaningful features + name meaningful features + MI of feature)
    print("worst: ",p.getStat()[sorted_d[0][0]])
    print("best: ", p.getStat()[sorted_d[-1][0]])
    print()
    best = probe.best_of_gen(parents)

    # ****
    #print("best of generation:")#print best genome/individual (with best FF in the current pop)
    #print(best)
    #print(probe.best_of_gen(parents).fitness)#print best genome/individual FF
    print()
    #print(p.getStat()[key])

    y = []
    #prediction for each observation/record of the best individual ([best.genome[i]])[0]) = ith observation)
    #len is about the number of genes in that genome (usually static)
    for i in range(len(best.genome)):
        y.append(classifier.predict([best.genome[i]])[0])

    #ch put the best individual in an array structure (1 el for each obs)
    ch = np.array(best.genome)

    print("MUTUAL INFO FOR EACH FEATURE - BEST INDIVIDUAL OF CURRENT POPULATION")
    MI = []
    for i in range(num_genes):
        print(features[i])
        mi = drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True)
        print(mi)
        MI.append([features[i],mi])
        
    result.append([p.getStat()[sorted_d[0][0]],p.getStat()[sorted_d[-1][0]],best,MI])

print()