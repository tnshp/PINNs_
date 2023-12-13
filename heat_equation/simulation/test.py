import pandas as pd
import numpy as np

u = np.zeros((100,5,5))
u = np.reshape(u, (100,25))

w =np.linspace([10,0,0], [20,30,0],10 )

m = np.ogrid[0:5, 5:10]
print(m)

