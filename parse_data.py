import glob
import pandas as pd
import os





score_files = ["unique_score1", "unique_score2", "cosine_score",  "overlap_score", "1entropy", "2entropy"]

celeba_valids = ["celeba", "male", "female", "noglass",  "nobeard", "nohat",  "nobeardnohat", "noglassnosmilenotie"]
#1 get all the potential dirs that matter
#2 in each of those dirs, look for the various scores
#3 make a dictionary of directory and scores
#4 make a pandas table of the parsed directory and each score
 
all_dirs = glob.glob(f"/usr/WS2/olson60/research/latentclr/experiments2/archive-eccv/full*")#k16_a3_usewTrue_enc-resnet50_losssimclr_oweight0.1*/colat*/20*/{file_type}_*")
all_dirs += glob.glob(f"/usr/WS2/olson60/research/latentclr/experiments2/full*")#/colat*/20*/")


exp_dirs =[]
for d in all_dirs:
    celebs = d.split("/")[-1]
    celeb1 = celebs.split("_")[1]
    celeb2 = celebs.split("_")[2]
    if celeb1 in celeba_valids and celeb2 in celeba_valids:
        more_exps = glob.glob(d + "/*/20*/")

        exp_dirs += more_exps

exps = {}

for d in exp_dirs:
    files = os.listdir(d)
    #if "final_model.pt" in files and "cosine_score.txt" not in files:
    #    print(d)
    cur_exp_results = {}
    for f in files:
        if f.endswith(".txt"): continue
        for sf in score_files:
            if sf in f:
                score = f.split("_")[-1]
                cur_exp_results[sf] = score
    if len(cur_exp_results.keys()) > 0:
        exps[d] = cur_exp_results


'''
/usr/WS2/olson60/research/latentclr/experiments2/
full_female_noglassnosmilenotie_k16_a3_usewTrue_enc-vit_losssimclr_oweight0.1_dre1.0_/
colat.generators.StyleGAN2Generator_colat.models.LinearConditional_colat.projectors.VitProjector4_trainprojFalsemae_pretrain_vit_base.pth/2022-08-01_19-57-54/

'''
exps_archiveless = {}
archive = "archive-eccv/"
for item in exps.keys():
    if archive in item:
        new = item.replace(archive, "")
        exps_archiveless[new] = exps[item]
    else:
        exps_archiveless[item] = exps[item]

for_panda = []
set_checker = {}

for item, values in exps_archiveless.items():
    print(item, len(item.split("/")))

    dir1 =  item.split("/")[7]
    dir2 =  item.split("/")[8]
    #celeba types
    celeba1 = dir1.split("_")[1]
    celeba2 = dir1.split("_")[2]
    dre = item.split("/")[7].split("dre")[1].split("_")[0]
    date = item.split("/")[9]
    #projector type
    proj_base = item.split("/")[7].split("_")[6][4:]
    if "resnet" in proj_base:
        if "advbn_augmix" in item:
            proj = "advbn"
        elif "att_classifier" in item:
            proj = "att"
        else:
            proj = "vanilla"
    else:
        if proj_base == "RN50":
            proj = "RN-clip"
        elif "ViT-B_32" in item:
            proj = "vit-clip"

        elif "14@336px" in item:
            proj = "vit-clip-large"
        elif "mae_pretrain_vit" in item:
            proj = "vit-mae"
        elif proj_base == "vit":
            proj = "vit"
        else:
            exit("bad projector directory")

    unique1 = values['unique_score1']
    unique2 = values['unique_score2']
    cosine_score = values['cosine_score']
    if celeba1[0] > celeba2[0]: 
        celeba1, celeba2 = celeba2, celeba1
        unique1, unique2 = unique2, unique1

    temp = celeba1+celeba2+proj+dre 
    if temp in set_checker: #and set_checker[temp] > cosine_score:
        #import pdb; pdb.set_trace()
        continue
    set_checker[temp] = cosine_score

    panda_row = [celeba1,celeba2,proj,dre, date,cosine_score , unique1, unique2]
    for_panda.append(panda_row)

print(for_panda)
df = pd.DataFrame(for_panda, columns = ["celeba1","celeba2","proj","dre", "date","cosine_score" , "unique1", "unique2"])
df.to_csv("parse_data2gen.csv", index=False)
#import pdb; pdb.set_trace()