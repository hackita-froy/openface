import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from collections import Counter


fname = '/home/michael/data/nli_faces/repsDbaseOld.pkl'
plt.ioff()

D = pkl.load(open(fname,'r'))

keys = D.keys()

ents = [k.split(r'/')[0] for k in keys]

C = Counter()
# C is a dict. key is entity - IE#####, value is number of counts this entity appeared, e.g:
# if C['IE12345']==12, entity IE12345 appeared 12 times

for e in ents:
    C[e] += 1

R = Counter()
# R is a dict. key is the number of appearances, value is the number of entities with that many appearences, e.g.:
# if R[12]==3, then there are 3 entities with 12 appearences

for v in C.values():
    R[v] += 1


# nApps = np.ndarray((len(R.keys())))
# nApps[:]=0
nApps = []

nEnts = []#np.copy(nApps)

for app in sorted(R.keys()):
    nApps.append(app)
    nEnts.append(R[app])

plt.close('all')
fig = plt.figure()
plt.plot(nApps,nEnts,'.-')
plt.xlabel('# of appearences')
plt.ylabel('# of entities')
ax = plt.gca()
ax.set_yscale('log')
plt.grid('on')

plt.show()