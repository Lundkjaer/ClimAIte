


import os
import random
import pandas as pd

#These list comprehensions turn a Grasshopper data tree into lists of lists
feature_names = ['u-value', 'infiltration', ...]
feature_bounds = [[0,100],[0,50],[0.1,0.75]]
feature_increments = [12,5,0.08]

PopulationPool = []

PopulationSize = 40
WinnerPercentage = 0.2
mutation_rate = 0.1
fitness_scores = []

generation_count = 0
max_count = 10

# Data to record
# [ [ generation, parent1, parent2, uniquename(gen-p1-p2), fitness_score, feature1, feature2, ..., featureN ] ]
# data_record = [[[]]]
data_record = pd.DataFrame(columns=['Generation', 'Parent1', 'Parent2', 'UniqueName', 'fitness Score', 'feature1', 'feature2'])

current_directory = os.getcwd()


BoundsRange = [] # a list of list of the ranges of the boundary conditions

# splitting the strings that describe the bounds ranges into lists of lists
# containing the start and end number
for i in range(len(feature_bounds)):
    # DesignBoundsPy[i] = (DesignBoundsPy[i][0].split())
    # DesignBoundsPy[i].pop(1) # this was when they were strings with '0 to 100'
    # make the strings into integers
    # for each in range(len(DesignBoundsPy[i])):
    #     DesignBoundsPy[i][each] = int(DesignBoundsPy[i][each])
    BoundsRange.append(feature_bounds[i][1] - feature_bounds[i][0])



allowed_genes = [] # a list of lists of eac allowed gene, created from the 
# increment sizes. This is used to initialise and create random mutations
#to ensure they are one of the valid bound increments.

#These two loops create a list of possible genes for each boundary condition.
for i in range(len(feature_bounds)):
    steps = []
    BoundsRange[i] / feature_increments[i]
    counter = 0
    while feature_bounds[i][0] + counter * feature_increments[i] < feature_bounds[i][1]:
        steps.append(round( (feature_bounds[i][0] + counter * feature_increments[i]) , 6))
        counter += 1
    # adding the top bound as some will not hit the top bound perfectly with their range division
    steps.append(feature_bounds[i][1])
    allowed_genes.append(steps)


def breeding(parents):
    '''
    This function returns a genome, generated from the parents pool.
    This current implementation allows for the two parents to be the same
    parent, in which case there will be a forced mutation on half the genes.
    In all cases there is a possibility for general random mutation,
    which will affect one gene and randomly pick another allowed value from
    the pool of allowed genes.
    
    returns a list, which a genome
    '''
    
    # randomly choose two parents
    parent1 = random.choice(parents)
    parent2 = random.choice(parents)
    child = []
    
    # for each gene, flip a coin and use that to select a gene from parent 1 or 2
    for i in range(len(parent1)): # for each gene pick from one parent
        coin = random.randint(0,1)
        if coin == 0:
            child.append(parent1[i])
        elif coin == 1:
            child.append(parent2[i])
        else:
            print('faulty coin')
    
    # incest mutation: If the two parents are the same, force a mutation
    if parent1 == parent2:
        for i in range(int(len(parent1)/2)): # mutate half the genes
            gene_no = random.randint(0,len(parent1)-1)
            child[gene_no] = random.choice(allowed_genes[gene_no])
            # This loop is set up such that maybe half the genes will mutate
            # but in other cases the same gene might be mutated
            # multiple times, leading to fewer overall mutations
            # this is an intentional part of the incest mutation.
    
    # random mutation based on the MutationRate input variable
    if random.randint(0,100) / 100 <= mutation_rate: #pick a random number, if that is lower than MutationRate than it mutates.
        gene_no = random.randint(0,len(child)-1)
        child[gene_no] = random.choice(allowed_genes[gene_no])
        
    return child



print(BoundsRange)
print(allowed_genes)

a  = 1/0



# put all into a loop instead?!

for i in range(max_count):

    # create first random PopulationPool
    if generation_count == 0: 
        #Generation 0
        PopulationPool = []
        # for each genome in populationsize, give it a collection of genes from the allowable genes pool.
        for i in range(PopulationSize):
            genome = []
            
            for each in range(len(feature_bounds)):
                genome.append(random.choice(allowed_genes[each]))
            PopulationPool.append(genome)
        
        generation_count += 1
        
        # convert list of lists to GH data tree structure as final step
        # PopulationPool = listToTree(PopulationPool)

    if generation_count >= max_count:
        a = 3 # placeholder



    # for subsequent loops we take the fitness score and associate it to the genomes. Then we sort them by their fitness.
    if generation_count > 0:
        for i in range(len(PopulationPool)):
            PopulationPool[i].append(fitness_scores[i])
        
        # sort by their fitess, which is now their last index item for each genome
        PopulationSorted = sorted(PopulationPool, key=lambda x: x[-1])
        
        
        # Parents breeding
        # the integer number of winners selected for breeding new generation
        WinnerNumber = int(round(PopulationSize * WinnerPercentage))
        
        Parents = [] #this becomes a list of list of features
        
        # for each winner add them to the parents list
        for i in range(WinnerNumber):
            parent = PopulationSorted[i]
            del parent[-1] # remoove the fitness value
            Parents.append(parent)
        
        
        # This here is the next generation, the children. The loop below populates 
        # the list calling the breeding() definition above. 
        children= []
        for i in range(PopulationSize):
            children.append( breeding(Parents) )
        
        # Then the new genome pool is the children.
        PopulationPool = children # new generation
        
        # Finally calling the helper function to convert to GH tree structure.
        # PopulationPool = listToTree(PopulationPool)

    data_record.concat()
    path = os.path.join(current_directory, '/data_records', f'record_{generation_count}')
    data_record.to_csv(path_or_buf = path)


def run_EP(genome):
    #runnign simulatin oand returning kwh result
    a = 4

def scoring_fitness():
    score = a * 5
