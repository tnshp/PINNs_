import pandas as pd
import numpy as np

data = pd.read_csv('heat_equation\DATA\data.csv', header=None).to_numpy()
data = np.reshape(data, (len(data), 50,50))
rand_matrix = np.random.rand(750,50,50) < 0.02 #should be zero or one 

u = list()
for i in range(len(data)):
    for j in range(50):
        for k in range(50):
            if rand_matrix[i,j,k] == True:
                u.append((i,j,k, data[i,j,k]))                             #search for condense function
            
sparse_data = pd.DataFrame(u)

sparse_data.to_csv('heat_equation\DATA\sparse_data.csv', header=False,index=False )