import h5py
import numpy as np
from collections import Counter
import pandas as pd

f = h5py.File('data/bkg_v1_edges.h5', 'r')

keys = []
output = []


def explore(data, output, k):
    ##If it's a group or file object we don't care, don't append that.
    if not isinstance(data, h5py._hl.files.File) and not isinstance(data, h5py._hl.group.Group):
        output.append(data)
    
    ## Don't store keys of group or file types
    else:
        try:
            k.pop()
        except:
            pass
    ## Try to go deeper. If we can't go deeper we reached the end and we stop.
    ## If we can go deeper, do it recursively
    try:
        keylist = list(data.keys())
        for key in keylist:
            k.append(key)
            explore(data[key], output, k)
            
    except:
        pass
    
explore(f, output, keys)

values =[]
for o in output:
    #print(f"Dimensions: {o.ndim}\nShape: {o.shape}\nSize: {o.size}\nBytes: {o.nbytes}")
    values.append(o[0:len(o)])
    
data = sorted(list(zip(values[0], values[3])), key = lambda x: x[1])
connections = Counter(i for i in data)
#connections = sorted(weight_counts, key = lambda x: x[1])



#connections = sorted(set(data), key = lambda x: x[1])
nya = []
for i in zip(connections.keys(), connections.values()):
    tmp = [i[0][0], i[0][1], i[1]]
    nya.append(tmp)
    
pop_names = ['e5Rbp4',
             'e23Cux2',
             'e4Scnn1a',
             'i6Htr3a',
             'e4other',
             'i4Sst',
             'e4Nr5a1',
             'i5Sst',
             'i4Pvalb',
             'i6Sst',
             'i6Pvalb',
             'i23Pvalb',
             'i23Htr3a', 
             'e4Rorb',
             'i5Htr3a',
             'i5Pvalb',
             'i23Sst',
             'i4Htr3a', 
             'e6Ntsr1', 
             'i1Htr3a',
             'e5noRbp4']
    
df = pd.DataFrame(nya)
for i in range(len(df[1])):
    df[1][i] = pop_names[df[1][i]]


