import glob



all_dirs = glob.glob(f"/usr/WS2/olson60/research/latentclr/experiments2/full*k16_a3*_usewTrue_enc-RN50_losssimclr_oweight0.1*dre1.0_/colat*/20*/final_model.pt")


celebas = ["toon","celebahq","anime","myanime","metfacesmall","disney","dog", "cat"]

celeb_dirs = []
add = True
for d in all_dirs:
    for c in celebas:
        if c in d: 
            add = False
            break

    if add and "final_model.pt" in d:
        celeb_dirs.append(d)
    add = True



cut = "final_model.pt"
#import pdb; pdb.set_trace()
gpu = 0
print("{ ", end="")
for i, item in enumerate(celeb_dirs):
    if i % (int(len(celeb_dirs)/4)) == 0 and i > 1: 
        print(" } & ")
        print("{ ", end="")

        gpu+=1
        if gpu > 3: gpu = 3

    print(" ./do_eval_gpu.sh 2eval.py ", gpu, " ", item[:-len(cut)], end=" ; ")
    #exit()
print(" } & ")



'''classes = ["celeba", "male", "female", "noglass", "nobeard ", "nohat", "nobeardnohat", "noglassnosmilenotie"]
tuples = []
for i in range(len(classes)):
    for j in range(i+1,len(classes)):
        tuples.append((classes[i], classes[j]))

for i in range(0,len(tuples),2):
    print(f"./submit_pbatch.sh {tuples[i][0]} {tuples[i][1]} {tuples[i+1][0]} {tuples[i+1][1]}")
'''
