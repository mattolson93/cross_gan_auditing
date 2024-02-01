import glob
import pandas as pd
import os

from itertools import combinations


checkfile = "sample_0.png"

all_dirs = glob.glob("./experiments/*/*/*/"+checkfile)

'''for gpu, exp in [(0, "voynov"),(1, "hessian"),(0, "jacobian"),(1, "conv1")]:

    hessian_dirs = []
    for d in all_dirs:
        if exp in d:
            hessian_dirs.append(d[:-len(checkfile)]) 

    hessian_dirs.sort()

    combos = list(combinations(hessian_dirs,2))


    print(" { ", end=" ")
    for n in combos:  print(f" ./do_eval_eval.sh eval_eval.py {gpu} {n[0]} {n[1]} ; ", end=" ")
    print(" } & ")

'''
#gpu = 0
#for gpu, exp in [(0, "voynov"),(1, "hessian"),(0, "jacobian"),(1, "conv1")]:
exp = "jacobian"
hessian_dirs = []
for d in all_dirs:
    if exp in d:
        hessian_dirs.append(d[:-len(checkfile)]) 

hessian_dirs.sort()

combos = list(combinations(hessian_dirs,2))



prevgpu = 0
print(" { ", end=" ")
for i, n in enumerate(combos):
    gpu = int((4*i)/len(combos))
    newgpu = prevgpu!=gpu
    print(f" ./do_eval_eval.sh eval_eval.py {gpu} {n[0]} {n[1]} ; ", end=" ")
    if newgpu: print(" } & \n {", end=" ")
    prevgpu = gpu
print("}")