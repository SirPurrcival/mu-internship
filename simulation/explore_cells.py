import numpy as np
import pylab as plt

import IPython

CELLS = np.load('cells.npy', allow_pickle=True).item()

types = CELLS.keys()

colors = plt.cm.Spectral(np.linspace(0, 1, len(types)))
colors = ['green', 'red', 'blue', 'green', 'yellow', 'red', 'blue', 'green', 'yellow', 'red', 'blue', 'green', 'yellow', 'red', 'blue', 'green', 'yellow']
plt.figure(figsize=(5, 5))
k = 0
for type in CELLS:
    for x in CELLS[type]:
        plt.scatter(x['E_L'], x['g'], color=colors[k])
    k += 1

plt.tight_layout()
plt.show()