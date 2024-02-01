import glob
import pandas as pd
import os





score_files = ["unique", "cosine",  "overlap_score", "1entropy", "2entropy"]

celeba_valids = ["celeba", "male", "female", "noglass",  "nobeard", "nohat",  "nobeardnohat", "noglassnosmilenotie"]
#1 get all the potential dirs that matter
#2 in each of those dirs, look for the various scores
#3 make a dictionary of directory and scores
#4 make a pandas table of the parsed directory and each score
 
all_dirs = glob.glob(f"/usr/WS2/olson60/research/latentclr/experiments/conv*")
all_dirs += glob.glob(f"/usr/WS2/olson60/research/latentclr/experiments/full*")

exp_dirs =[]
for d in all_dirs:
    celebs = d.split("/")[-1]
    celeb1 = celebs.split("_")[1]
    if celeb1 in celeba_valids:
        more_exps = glob.glob(d + "/*/20*/")

        exp_dirs += more_exps

exp_rows = []

for d in exp_dirs:
    files = os.listdir(d)
    loss_func = d.split("_")[4]
    celeb1=None
    for c in celeba_valids: 
        if c in d: celeb1=c
    #if "final_model.pt" in files and "cosine_score.txt" not in files:
    #    print(d)
    cur_exp_results = {}
    for f in files:
        if f.endswith(".txt") or f.endswith(".png"): continue
        for sf in score_files:
            if sf in f:
                if len(f.split("_")) == 4: continue
                celeb2, filetype, n_overlap, scorex, score = f.split("_")

                exp_rows.append([loss_func, celeb1, celeb2, filetype, n_overlap, scorex, score])
    
#breakpoint()


df = pd.DataFrame(exp_rows, columns = ["loss","celeb1", "celeb2", "filetype", "n_overlap", "scorex", "score"])
df.to_csv("temp.csv", index=False)
#import pdb; pdb.set_trace()