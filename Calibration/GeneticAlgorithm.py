


import os
import random
import pandas as pd
import numpy as np

from fitness_function import fitness_calc
# from run import run_EP

#These list comprehensions turn a Grasshopper data tree into lists of lists
feature_names = ['u-value', 'infiltration', ...]
feature_bounds = [[0,100],[0,50],[0.1,0.75]]
feature_increments = [12,5,0.08]
# or list
feats = [['u-value', [0,100], 12], ['infiltration', [0.1,0.6], 0.04]]
features_number = 2 # make calculated TODO
# name_fields_number = 4 # make calculated TODO
# calculated_fields_number = 1 # make calculated TODO
names_fields = ['Generation', 'Parent1', 'Parent2', 'Unique Name']
calculated_fields = ['Fitness Score', 'kWh result']

# PopulationPool = []

PopulationSize = 40
WinnerPercentage = 0.2
mutation_rate = 0.1

generation_count = 0
max_count = 10

# Data to record
# [ [ generation, parent1, parent2, uniquename(gen-p1-p2), fitness_score, feature1, feature2, ..., featureN ] ]
# data_record = [[[]]]
column_names = ['Generation', 'Parent1', 'Parent2', 'Unique Name', 'Fitness Score', 'feature1', 'feature2' ] # also add kWh output as a column

data_record = pd.DataFrame(columns=column_names)

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



def breeding():
    '''
    This function returns a genome, generated from the parents pool.
    This current implementation allows for the two parents to be the same
    parent, in which case there will be a forced mutation on half the genes.
    In all cases there is a possibility for general random mutation,
    which will affect one gene and randomly pick another allowed value from
    the pool of allowed genes.
    
    returns a list, which a genome
    '''
    WinnerNumber = int(round(PopulationSize * WinnerPercentage))

    prev_gen_pool = data_record[ data_record['Generation'] == generation_count ] # select current generation
    prev_gen_pool = prev_gen_pool.sort_values(by=['Fitness Score']) # sort by fitness
    parent_pool = prev_gen_pool.iloc[:WinnerNumber] # select only the top winner pool

    for new_child in range(PopulationSize):
        # randomly choose two parents
        parent1 = parent_pool.sample(n=1)
        parent2 = parent_pool.sample(n=1)
        
        
        child = []
        # for each gene, flip a coin and use that to select a gene from parent 1 or 2
        for i in range(features_number): # for each gene pick from one parent
            coin = random.randint(0,1)
            if coin == 0:
                child.append(parent1.iloc[0,len(names_fields) + len(calculated_fields) + i])
            elif coin == 1:
                child.append(parent2.iloc[0,len(names_fields) + len(calculated_fields) + i])
            else:
                print('faulty coin')
        
        # incest mutation: If the two parents are the same, force a mutation
        if parent1['Unique Name'] == parent2['Unique Name']:
            for i in range(int(len(parent1)/2)): # mutate half the genes
                gene_no = random.randint(0,len(parent1)-1)
                child[gene_no] = random.choice(allowed_genes[gene_no])
                # This loop is set up such that maybe half the genes will mutate
                # but in other cases the same gene might be mutated
                # multiple times, leading to fewer overall mutations
                # this is an intentional part of the incest mutation.
        
        # random mutation based on the MutationRate input variable
        if random.randint(0,101) / 100 <= mutation_rate: #pick a random number, if that is lower than MutationRate than it mutates.
            gene_no = random.randint(0,len(child)-1)
            child[gene_no] = random.choice(allowed_genes[gene_no])
        
        name_fields_list = [generation_count,
                          parent1['Unique Name'],
                          parent2['Unique Name'],
                          f'{generation_count}-{new_child}'
                          ]
        calc_field_list = [np.NaN for i in range(len(calculated_fields))]

        child = name_fields_list + calc_field_list + child
        

        data_record = pd.concat([data_record,
                                pd.DataFrame([child],
                                    columns=column_names )
                                    ])
    
    # Add to dataframe - genration +1
    # return child
    data_record = data_record.reset_index(drop=True)

def run_generation():
    # this below is hte same in hte main loop - could be a function instead
    current_gen = data_record[ data_record['Generation'] == generation_count ].reset_index(drop=True)
    # current_gen = current_gen.reset_index(drop=True)
    assert current_gen.shape[0] == PopulationSize

    for each in range(PopulationSize):
        kwh_result = run_EP(current_gen.iloc[[each]])
        current_gen.at[each, 'kWh result'] = kwh_result

    for each in range(PopulationSize):
        fit_result = fitness_calc(current_gen.at[each, 'kWh result'])
        current_gen.at[each, 'Fitness Score'] = fit_result


print(BoundsRange)
print(allowed_genes)
print(data_record)

a  = 1/0



# put all into a loop instead?!

for i in range(max_count):

    # create first random PopulationPool
    if generation_count == 0: 

        # PopulationPool = []
        # for each genome in populationsize, give it a collection of genes from the allowable genes pool.
        for i in range(PopulationSize):
            first_genome = []
            
            for each in range(len(feature_bounds)):
                genome.append(random.choice(allowed_genes[each]))
            PopulationPool.append(genome)

            name_fields_list = [generation_count,
                    'no parent',
                    'no parent',
                    f'{generation_count}-{i}'
                    ]
            calc_field_list = [np.NaN for i in range(len(calculated_fields))]

            first_genome = name_fields_list + calc_field_list + first_genome
            
            data_record = pd.concat([data_record,
                                    pd.DataFrame([first_genome],
                                        columns=column_names )
                                        ])
            

        
        data_record = data_record.reset_index(drop=True)

        run_generation()

    # for subsequent loops we take the fitness score and associate it to the genomes. Then we sort them by their fitness.
    elif generation_count > 0:

        breeding()
        run_generation()

    path = os.path.join(current_directory, '/data_records', f'record_{generation_count}')
    data_record.to_csv(path_or_buf = path)

    generation_count += 1


def run_EP(genome):
    #runnign simulatin oand returning kwh result

    # genome is single index df
    assert genome.shape[0] == 1

    gen_features = list(genome.loc[0, [feature_names]])


    # Energyplus

    kWh = 0
    for i in range(len(gen_features)):
        kWh += gen_features[i]

    return kWh

    # return kWh

