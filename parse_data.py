import glob

all_dirs = glob.glob("/usr/WS2/olson60/research/latentclr/experiments2/full*k16_a3_usewTrue_enc-resnet50_losssimclr_oweight0.1*/colat*/20*/unique_score2_*")


celebas = ["toon","celebahq","anime","myanime","metfacesmall","disney","dog", "cat"]

celeb_dirs = []
add = True
for d in all_dirs:
    for c in celebas:
        if c in d: 
            add = False
            break

    if add:
        celeb_dirs.append(d)
    add = True


cut = "/usr/WS2/olson60/research/latentclr/experiments2/full_"



for item in celeb_dirs:
    end = item[len(cut):]
    gen1 = end.split("_")[0]
    gen2 = end.split("_")[1]
    score = end.split("_")[-1]
    dre = end.split("_")[8][3:]
    model_type = "normal"
    if "advbn" in end:
        model_type = "advbn"
    if "att_classifier" in end:
        model_type = "att"


    print(gen1, gen2, model_type, dre, score)
    #exit()

print(len(celeb_dirs))
print(len(all_dirs))
