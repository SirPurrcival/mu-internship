import h5py
import numpy as np

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
    
# data = dict()
# for k, v in zip(keys, values):
#     data[k] = v
    
# np.unique(values[0]).size
# np.unique(values[3]).size
