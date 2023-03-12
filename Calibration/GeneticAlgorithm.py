


import os
import random
import pandas as pd
import numpy as np

from fitness_function import fitness_calc
# from run import run_EP


names_fields = ['Generation', 'Parent1', 'Parent2', 'Unique Name']
calculated_fields = ['Fitness Score', 'kWh result']

#               'name'      [bounds]    step-size
feature_meta = [['idf.People.People_5a9ee43c.People_per_Floor_Area', [0.02,0.3], 0.01],
        ['idf.ElectricEquipment.Equipment_d01c6929.Watts_per_Zone_Floor_Area', [1,15], 1],
        ['idf.ZoneInfiltration:DesignFlowRate.Infiltration_d1fbaf75.Flow_Rate_per_Exterior_Surface_Area', [0,0.0003], 0.00002],
        ['idf.WindowMaterial:SimpleGlazingSystem.PH_Window.UFactor', [0.4,1.4], 0.1],
        ['idf.WindowMaterial:SimpleGlazingSystem.PH_Window.Solar_Heat_Gain_Coefficient', [0.3,0.9], 0.1],
        ['idf.Material:NoMass.Typical Insulation-R50.Thermal_Resistance', [3,20], 1],
        ['idf.Material.Generic Gypsum Board.Specific_Heat', [500,3000], 100],
        ['idf.ZoneHVAC:IdealLoadsAirSystem.Z1_Ground_Floor_48970ba6 Ideal Loads Air System.Sensible_Heat_Recovery_Effectiveness', [0.4,0.95], 0.2]
                ]


PopulationSize = 20
WinnerPercentage = 0.2
mutation_rate = 0.2

generation_counter = 0   #always 0
max_count = 10

feature_names = [x[0] for x in feature_meta]
feature_bounds = [x[1] for x in feature_meta]
feature_increments = [x[2] for x in feature_meta]

column_names = names_fields + calculated_fields + feature_names
data_record = pd.DataFrame(columns=column_names)


current_directory = os.getcwd()


BoundsRange = [] # a list of list of the ranges of the boundary conditions
# splitting the strings that describe the bounds ranges into lists of lists
# containing the start and end number
for i in range(len(feature_bounds)):
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


def breeding(data_record):
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

    prev_gen_pool = data_record[ data_record['Generation'] == generation_counter - 1 ] # select previous generation
    prev_gen_pool = prev_gen_pool.sort_values(by=['Fitness Score']) # sort by fitness
    parent_pool = prev_gen_pool.iloc[:WinnerNumber] # select only the top winner pool

    for new_child in range(PopulationSize):
        # randomly choose two parents
        parent1 = parent_pool.sample(n=1)
        parent2 = parent_pool.sample(n=1)
        
        
        child = []
        # for each gene, flip a coin and use that to select a gene from parent 1 or 2
        for i in range(len(feature_names)): # for each gene pick from one parent
            coin = random.randint(0,1)
            if coin == 0:
                child.append(parent1.iloc[0,len(names_fields) + len(calculated_fields) + i])
            elif coin == 1:
                child.append(parent2.iloc[0,len(names_fields) + len(calculated_fields) + i])
            else:
                print('faulty coin')
        
        # incest mutation: If the two parents are the same, force a mutation
        if parent1['Unique Name'].iloc[0] == parent2['Unique Name'].iloc[0]:
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
        
        name_fields_list = [generation_counter,
                          parent1['Unique Name'].iloc[0],
                          parent2['Unique Name'].iloc[0],
                          f'{generation_counter}-{new_child}'
                          ]
        calc_field_list = [np.NaN for i in range(len(calculated_fields))]

        child = name_fields_list + calc_field_list + child
        

        data_record = pd.concat([data_record,
                                pd.DataFrame([child],
                                    columns=column_names )
                                    ])
        
    data_record = data_record.reset_index(drop=True)
    return data_record



def run_generation(data_record):
    # this below is hte same in hte main loop - could be a function instead
    current_gen = data_record[ data_record['Generation'] == generation_counter ]#.reset_index(drop=True)
    assert current_gen.shape[0] == PopulationSize
    indices = current_gen.index

    # replace. Use index


    for i, index in enumerate(indices):
        kwh_result = run_EP(current_gen.iloc[[i]])
        # current_gen.at[i, 'kWh result'] = kwh_result
        data_record.at[index, 'kWh result'] = kwh_result



    for i, index in enumerate(indices):
        fit_result = fitness_calc(data_record.at[index, 'kWh result'])
        # fit_result = fitness_calc(current_gen.at[i, 'kWh result'])
        # current_gen.at[i, 'Fitness Score'] = fit_result
        data_record.at[index, 'Fitness Score'] = fit_result

    return data_record

def run_EP(genome):
    #runnign simulatin oand returning kwh result
    # genome is single index df
    assert genome.shape[0] == 1
    # gen_features = list(genome.loc[0, feature_names])
    df = genome[feature_names]
    gen_features = list(df.iloc[0])

    # Energyplus

    kWh = 0
    for i in range(len(gen_features)):
        kWh += gen_features[i]

    return round(kWh, 2)

# Main loop
for i in range(max_count):

    # create first random PopulationPool
    if generation_counter == 0:

        # PopulationPool = []
        # for each genome in populationsize, give it a collection of genes from the allowable genes pool.
        for i in range(PopulationSize):
            first_genome = []
            
            for each in range(len(feature_names)):
                first_genome.append(random.choice(allowed_genes[each]))

            name_fields_list = [generation_counter,
                    'no parent',
                    'no parent',
                    f'{generation_counter}-{i}'
                    ]
            calc_field_list = [np.NaN for i in range(len(calculated_fields))]

            first_genome = name_fields_list + calc_field_list + first_genome
            
            data_record = pd.concat([data_record,
                                    pd.DataFrame([first_genome],
                                        columns=column_names )
                                        ])
            

        
        data_record = data_record.reset_index(drop=True)

        data_record = run_generation(data_record)

    # for subsequent loops we take the fitness score and associate it to the genomes. Then we sort them by their fitness.
    elif generation_counter > 0:

        data_record = breeding(data_record)
        data_record = run_generation(data_record)
        # run_generation()

    path_record_name = os.path.join(current_directory, 'data_records', f'record_{generation_counter}')
    data_record.to_csv(path_or_buf = path_record_name)

    generation_counter += 1




