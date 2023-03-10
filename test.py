import pandas as pd
import numpy as np

column_names = ['Generation', 'Parent1', 'Parent2', 'UniqueName', 'feature1', 'feature2', 'Fitness Score'] # also add kWh output as a column

data_record = pd.DataFrame(columns=column_names)

generation_count = 1

data_record = pd.concat([data_record, pd.DataFrame([[generation_count, 2, 15, f'{generation_count}-2-15', 4.3, 3, np.NaN]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[3, 2, 15, f'{generation_count}-2-15', 4.3, 3, 84]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[generation_count, 2, 15, f'{generation_count}-2-15', 4.3, 3, np.NaN]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[2, 2, 15, f'{generation_count}-2-15', 4.3, 3, 64]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[generation_count, 2, 15, f'{generation_count}-2-15', 4.3, 3, np.NaN]], columns=column_names ) ])
# data_record = pd.concat([data_record, pd.DataFrame([[5, 2, 15, f'{generation_count}-2-15', lambda a : a + 10 ]], columns=column_names ) ])
data_record = data_record.reset_index(drop=True)

print(data_record.head())
print(data_record.shape)

data_record.loc[4, 'Fitness Score'] = 23
data_record.loc[2, 'Fitness Score'] = 53
data_record.loc[0, 'Fitness Score'] = 46

print(data_record.head())
print(data_record.shape)

win_num = 2

prev_gen_pool = data_record[ data_record['Generation'] == generation_count ]
prev_gen_pool = prev_gen_pool.sort_values(by=['Fitness Score'])
parent_pool = prev_gen_pool.iloc[:win_num]

child = [3, 2, 15, f'{generation_count}-2-15', 4.3, 3, 84]

# parent_pool = pd.concat([parent_pool,
#                         pd.DataFrame([child],
#                             columns=column_names )
#                             ])

print(prev_gen_pool.head())
print(parent_pool.head())

parent1 = parent_pool.sample(n=1)
# print(parent1[4 + 1 + 1])
print(parent1)
features = ['feature1', 'feature2']
# print(parent1.iloc[0,4+1])
# print(parent1.shape[1])
# print('shape is ', parent_pool.shape)
# # print((parent1[['feature1', 'feature2']]))
# print(list(parent1.at[0, [features]]))
df = parent1[features]
gen_features = list(df.iloc[0])
print(gen_features)
name = parent1['UniqueName'].iloc[0]
print(name)
# # sort on fitness

# for each in prev_gen_pool[ prev_gen_pool['Generation'] == generation_count ]:
#     print(each)


# from Calibration.fitness_function import fitness_calc

# print(fitness_calc(500))