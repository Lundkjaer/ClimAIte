import pandas as pd
import numpy as np

column_names = ['Generation', 'Parent1', 'Parent2', 'UniqueName', 'feature1', 'feature2', 'Fitness Score'] # also add kWh output as a column

data_record = pd.DataFrame(columns=column_names)

generation = 1

data_record = pd.concat([data_record, pd.DataFrame([[generation, 2, 15, f'{generation}-2-15', 4.3, 3, np.NaN]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[3, 2, 15, f'{generation}-2-15', 4.3, 3, 84]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[generation, 2, 15, f'{generation}-2-15', 4.3, 3, np.NaN]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[2, 2, 15, f'{generation}-2-15', 4.3, 3, 64]], columns=column_names ) ])
data_record = pd.concat([data_record, pd.DataFrame([[generation, 2, 15, f'{generation}-2-15', 4.3, 3, np.NaN]], columns=column_names ) ])
data_record = data_record.reset_index(drop=True)

print(data_record.head())
print(data_record.shape)

data_record.loc[4, 'Fitness Score'] = 23
data_record.loc[2, 'Fitness Score'] = 53
data_record.loc[0, 'Fitness Score'] = 46

print(data_record.head())
print(data_record.shape)

win_num = 2

prev_gen_pool = data_record[ data_record['Generation'] == generation ]
prev_gen_pool = prev_gen_pool.sort_values(by=['Fitness Score'])
parent_pool = prev_gen_pool.iloc[:win_num]


print(prev_gen_pool.head())
print(parent_pool.head())

# sort on fitness

