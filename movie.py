import pandas as pd
import numpy as np


nameset= pd.read_excel('nameset.xlsx',)
nameset=np.array(nameset)
name = pd.read_excel('name.xlsx')
name = np.array(name)
name1 = name[:,0]



c=[]
for i in range(len(name1)):

    for j in range(len(nameset)):
        if name1[i] in nameset[j]:
            c.append(j)
        else:
            continue
import pickle
with open('ccc','wb') as mysavedata:
    pickle.dump(c, mysavedata)
