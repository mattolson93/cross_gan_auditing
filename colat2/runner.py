import os
import lpips

import hydra
import torch
import torch.nn as nn
from torch.nn import functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torchvision.transforms as transforms

from colat2.evaluator import Evaluator
from colat2.evaluator import Evaluator2
from colat2.projectors import Projector
from colat2.trainer import Trainer
from colat2.visualizer import Visualizer

import copy

from colat.utils.net_utils import create_dre_model
from colat.utils.net_utils import create_bce_model

def train(cfg: DictConfig) -> None:
    """Trains model from config

    Args:
        cfg: Hydra config

    """
    # Device
    device = get_device(cfg)

    
    # Model
    
    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)
    #import pdb; pdb.set_trace()
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    cfg.model['size'] = generator2.n_latent()
    model2: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)


    projector: Projector = instantiate(cfg.projector, img_preprocess=cfg.imgproc, min_resolution=min(generator.resolution, generator2.resolution)).to(device)
    #projector.resize = transforms.CenterCrop(224)
    #projector.resnet_resize = transforms.CenterCrop(224) #transforms.Resize(128)
   

    params_list = list(model.parameters()) + list(model2.parameters())
    if cfg.train_projector: params_list +=  list(projector.parameters())

    if "do_trans" in cfg.exps.keys():
        trans_model = TransNet().to(device)
        params_list += list(trans_model.parameters())
    else:
        trans_model = None

    optimizer: torch.optim.Optimizer = instantiate(  cfg.hparams.optimizer, params_list    )
    scheduler = instantiate(cfg.hparams.scheduler, optimizer)

    # Paths
    save_path = os.getcwd() if cfg.save else None
    checkpoint_path = (
        hydra.utils.to_absolute_path(cfg.checkpoint)
        if cfg.checkpoint is not None
        else None
    )

    # Tensorboard
    if cfg.tensorboard:
        # Note: global step is in epochs here
        writer = SummaryWriter(os.getcwd())
        # Indicate to TensorBoard that the text is pre-formatted
        text = f"<pre>{OmegaConf.to_yaml(cfg)}</pre>"
        writer.add_text("config", text)
    else:
        writer = None

    if "ffhqu-r" in cfg['generator']['class_name'] and "ffhqu-t" in cfg['generator2']['class_name']:
        raise ValueError("must have StyleGan3-t as generator and StyleGan3-r as generator2")

    #breakpoint()
    do_pool = (cfg.projector.layers == 4) and "resnet" in cfg.projector.name 
    if cfg.dre_lamb >0:

        dre_model = create_dre_model(projector.get_size(), do_pool) 
        resize_string = cfg.imgproc if cfg.imgproc != "resize" else ""
        dre_path = os.path.join( hydra.utils.get_original_cwd(), 'dre_models',cfg.projector.name + "_" + str(cfg.projector.layers) +  "_" +  cfg.projector.load_path + "_" + cfg['generator']['class_name']+ "_" +cfg['generator2']['class_name'] + resize_string + ".pt")
        if not os.path.exists(dre_path):
            if "ffhqu-t" in cfg['generator']['class_name'] and "ffhqu-r" in cfg['generator2']['class_name']:
                dre_model = train_dre_without_generators(dre_model,  projector, dre_path)
            else:
                dre_model = train_dre(dre_model, generator, generator2, projector)
            torch.save(dre_model.state_dict(), dre_path)
        else:
            dre_model.load_state_dict(torch.load(dre_path))

    elif cfg.dre_lamb < 0:
        dre_model = create_bce_model(projector.get_size(), do_pool) 
        bce_path = os.path.join( hydra.utils.get_original_cwd(), 'bce_models',cfg.projector.name + "_" + str(cfg.projector.layers) +  "_" +  cfg.projector.load_path + "_" + cfg['generator']['class_name']+ "_" +cfg['generator2']['class_name'] + ".pt")
        if not os.path.exists(bce_path):
            dre_model = train_bce(dre_model, generator, generator2, projector)
            torch.save(dre_model.state_dict(), bce_path)
        else:
            dre_model.load_state_dict(torch.load(bce_path))
    else:
        dre_model = None

    loss_fn: torch.nn.Module = instantiate(cfg.loss, k=cfg.k, size=projector.get_size()*16 if do_pool else projector.get_size()).to(device)

    # Trainer init
   

    cfg.n_dirs = list(range(cfg.k))
    #import pdb; pdb.set_trace()
    invert_z_file = os.path.join( hydra.utils.get_original_cwd(), 'invertedpoints',cfg['generator']['class_name']+ "_" +cfg['generator2']['class_name'] + ".pt")
    cfg.n_samples = 5
    '''if os.path.exists(invert_z_file):
        zs = torch.load(invert_z_file, map_location=device)
        z1 = zs[:cfg.n_samples,0]
        z2 = zs[:cfg.n_samples,1]
    
        if generator.w_primary:  z1 = generator.convert_z2w(z1)
        if generator2.w_primary: z2 = generator2.convert_z2w(z2)

    else:
        z1, z2 = invert_2_gens(generator, generator2, cfg.n_samples)
        torch.save(torch.stack([z1,z2],dim=1),invert_z_file)'''
    #from sklearn.metrics.pairwise import cosine_similarity as cos
    #cos(model.state_dict()['params'].cpu().numpy())

    visualizer = Visualizer(
        model=model,
        generator=generator,
        device=device,
        n_samples=cfg.n_samples,
        #n_samples=z1,
        n_dirs=cfg.n_dirs,
        alphas=cfg.alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        image_size=cfg.image_size,
        gen_ind=1,
    )


    visualizer2 = Visualizer(
        model=model2,
        generator=generator2,
        device=device,
        n_samples=cfg.n_samples,
        #n_samples=z2,
        n_dirs=cfg.n_dirs,
        alphas=cfg.alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        image_size=cfg.image_size,
        gen_ind=2,
    )
    
    trainer = Trainer(
        model=model,
        model2=model2,
        loss_fn=loss_fn,
        optimizer=optimizer,
        generator=generator,
        generator2=generator2,
        projector=projector,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        overlap_k=cfg.overlap_k,
        device=device,
        visualizer=visualizer,
        visualizer2=visualizer2,
        eval_freq=cfg.eval_freq,
        eval_iters=cfg.eval_iters,
        scheduler=scheduler,
        grad_clip_max_norm=cfg.hparams.grad_clip_max_norm,
        writer=writer,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        mixed_precision=cfg.mixed_precision,
        train_projector=cfg.train_projector,
        feed_layers=cfg.feed_layers,
        dre_model=dre_model,
        dre_lamb=cfg.dre_lamb,
        extra_stuff=cfg.exps,
        trans_model=trans_model
    )

    evaluator = Evaluator(
        model=model,
        model2=model2,
        generator=generator,
        generator2=generator2,
        device=device,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        num_unique_directions=(cfg.k - cfg.overlap_k),
        total_directions=cfg.k,
        att_model_path=os.path.join(hydra.utils.get_original_cwd(),"att_classifier.pt")
    )
    

    # Launch training process
    trainer.train()

    #import pdb; pdb.set_trace()
    '''overlap_direction_scores, cosines, unique_score1, unique_score2 = evaluator.evaluate()
    with open('overlap_score.txt', 'w') as f:       f.write(f"{overlap_direction_scores}")
    with open(f"overlap_score_{overlap_direction_scores}", 'w') as f:  f.write("")
    with open('cosine_score.txt', 'w') as f:       f.write(f"{cosines}")
    with open(f"cosine_score_{cosines}", 'w') as f:  f.write(f"")
    with open('unique_score1.txt', 'w') as f:       f.write(f"{unique_score1}")
    with open(f"unique_score1_{unique_score1}", 'w') as f:  f.write("")
    with open('unique_score2.txt', 'w') as f:       f.write(f"{unique_score2}")
    with open(f"unique_score2_{unique_score2}", 'w') as f:  f.write("")'''

    #final visual
    visualizer.visualize()
    visualizer2.visualize()
    clamp_val = 1.00

    visualizer.visualize(clamp_val=clamp_val)
    visualizer2.visualize(clamp_val=clamp_val)


def evaluate(cfg: DictConfig) -> None:
    """Evaluates model from config

    Args:
        cfg: Hydra config
    """
    # Device
    #import pdb; pdb.set_trace()
    device = get_device(cfg)
    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)
    
    #projector: torch.nn.Module = instantiate(cfg.projector).to(device)
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    cfg.model['size'] = generator2.n_latent()
    model2: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)

    # Preload model
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model2.load_state_dict(checkpoint["model2"])
    #import pdb; pdb.set_trace()
    #if checkpoint["projector"] is not None:
    #    projector.load_state_dict(checkpoint["projector"])

    evaluator = Evaluator(
        model=model,
        model2=model2,
        generator=generator,
        generator2=generator2,
        device=device,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        num_unique_directions=(cfg.k - cfg.overlap_k),
        total_directions=cfg.k,
        att_model_path=os.path.join(hydra.utils.get_original_cwd(),"att_classifier.pt")
    )
    #import pdb; pdb.set_trace()
    overlap_direction_scores, cosines, unique_score1, unique_score2 = evaluator.evaluate()
    with open('overlap_score.txt', 'w') as f:       f.write(f"{overlap_direction_scores}")
    with open(f"overlap_score_{overlap_direction_scores}", 'w') as f:  f.write("")
    with open('cosine_score.txt', 'w') as f:       f.write(f"{cosines}")
    with open(f"cosine_score_{cosines}", 'w') as f:  f.write(f"")
    with open('unique_score1.txt', 'w') as f:       f.write(f"{unique_score1}")
    with open(f"unique_score1_{unique_score1}", 'w') as f:  f.write("")
    with open('unique_score2.txt', 'w') as f:       f.write(f"{unique_score2}")
    with open(f"unique_score2_{unique_score2}", 'w') as f:  f.write("")

    '''
    score = evaluator.evaluate_entropy(model, generator)
    with open('1entropy.txt', 'w') as f:       f.write(f"{score}")
    with open(f"1entropy_{score}", 'w') as f:  f.write("")


    trunc_val = 1.0
    score = evaluator.evaluate_entropy(model, generator, trunc_val)
    with open(f'1entropy_t{trunc_val}.txt', 'w') as f:       f.write(f"{score}")
    with open(f"1entropy_t{trunc_val}_s{score}", 'w') as f:  f.write("")



    score = evaluator.evaluate_entropy(model2, generator2)
    with open('2entropy.txt', 'w') as f:       f.write(f"{score}")
    with open(f"2entropy_{score}", 'w') as f:  f.write("")


    trunc_val = 1.0
    score = evaluator.evaluate_entropy(model2, generator2, trunc_val)
    with open(f'2entropy_t{trunc_val}.txt', 'w') as f:       f.write(f"{score}")
    with open(f"2entropy_t{trunc_val}_s{score}", 'w') as f:  f.write("")'''


def evaluate_two_seperates(cfg: DictConfig, cfg2: DictConfig) -> None:
    """Evaluates model from config

    Args:
        cfg: Hydra config
    """
    # Device
    #import pdb; pdb.set_trace()
    device = get_device(cfg)
    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg2.generator).to(device)
    
    #projector: torch.nn.Module = instantiate(cfg.projector).to(device)
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    cfg.model['size'] = generator2.n_latent()
    model2: torch.nn.Module = instantiate(cfg2.model, k=cfg.k, batch_k=cfg.batch_k).to(device)

    # Preload model
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    ckp2_path = os.path.join(hydra.utils.get_original_cwd(), cfg.cfg2, cfg.checkpoint)
    checkpoint2 = torch.load(ckp2_path, map_location=device)
    model2.load_state_dict(checkpoint2["model"])

    if not isinstance(model.alpha, int): model.alpha = max(model.alpha)
    if not isinstance(model2.alpha, int): model2.alpha = max(model2.alpha)

    #import pdb; pdb.set_trace()
    #if checkpoint["projector"] is not None:
    #    projector.load_state_dict(checkpoint["projector"])

    evaluator = Evaluator2(
        model=model,
        model2=model2,
        generator=generator,
        generator2=generator2,
        device=device,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        num_unique_directions=(cfg.k - cfg.overlap_k),
        total_directions=cfg.k,
        att_model_path=os.path.join(hydra.utils.get_original_cwd(),"att_classifier.pt")
    )
    #import pdb; pdb.set_trace()
    cutoffs, cosses, unique_list1, unique_list2 = evaluator.evaluate()
    for c, cos, unique_score1, unique_score2 in zip(cutoffs, cosses, unique_list1, unique_list2):
        with open(f'{generator2.outclass}_cosine_{c}_score.txt', 'w') as f:       f.write(f"{cos}")
        with open(f"{generator2.outclass}_cosine_{c}_score_{cos}", 'w') as f:  f.write(f"")
        with open(f'{generator2.outclass}_unique_{c}_score1.txt', 'w') as f:       f.write(f"{unique_score1}")
        with open(f"{generator2.outclass}_unique_{c}_score1_{unique_score1}", 'w') as f:  f.write("")
        with open(f'{generator2.outclass}_unique_{c}_score2.txt', 'w') as f:       f.write(f"{unique_score2}")
        with open(f"{generator2.outclass}_unique_{c}_score2_{unique_score2}", 'w') as f:  f.write("")


def train_dre(dre_model, generator, generator2, projector):

    dre_optim = torch.optim.Adam( dre_model.parameters(),lr=1e-2)

    dre_model.train()
    projector.eval()
    generator.eval()
    generator2.eval()

    resize = projector.transform_preprocess

    with torch.no_grad():
        bs = 2
        try:
            while True: 
                img1 = generator( generator.sample_latent(bs*2))
                img2 = generator2(generator2.sample_latent(bs*2))
                imgs = torch.cat([resize(img1),resize(img2)],dim=0)
                feats = projector(imgs)
                bs*=2
                print("working batchsize: ", bs)
                if bs == 32: break
        except:
            print("optimal batch size found at: ", bs)
    
    iterations = 1000
    pbar = tqdm(total=iterations, leave=False)

    outlier_loss_list = []
    for i in range(iterations):
        dre_optim.zero_grad()
        with torch.no_grad():
            img1 = generator( generator.sample_latent(bs))
            torch.cuda.empty_cache()

            img2 = generator2(generator2.sample_latent(bs))
            torch.cuda.empty_cache()

            imgs = torch.cat([resize(img1),resize(img2)],dim=0)

            feats = projector(imgs)
            torch.cuda.empty_cache()
            
        dre_logit = dre_model(feats)
        
        dre_logit1 = dre_logit[:, 0]
        dre_logit2 = dre_logit[:, 1]

        inlier_loss = -torch.log(dre_logit1[:bs]).mean() - torch.log(dre_logit2[bs:]).mean()
        outlier_loss = dre_logit1[bs:].mean() + dre_logit2[:bs].mean()

        loss = inlier_loss + outlier_loss
        loss.backward()
        dre_optim.step()

        inlier_avg1 = dre_logit1[:bs].mean().item()
        inlier_avg2 = dre_logit2[bs:].mean().item()

        outlier_avg1 = dre_logit1[bs:].mean().item()
        outlier_avg2 = dre_logit2[:bs].mean().item()

        outlier_loss_list.append(outlier_loss.item())
        if (inlier_avg1 + inlier_avg2)/2 > 49 and i > 150: break
        pbar.update()
        pbar.set_postfix_str(f"in1 {inlier_avg1:.3f}, in2 {inlier_avg2:.3f}, o1 {outlier_avg1:.3f}, o2 {outlier_avg2:.3f}, iloss {inlier_loss.item():.3f}, oloss {outlier_loss.item():.3f}", refresh=False)
        torch.cuda.empty_cache()

    pbar.close()
    print("outlir losses:")
    for i in outlier_loss_list: print(i)
    return dre_model

from torchvision import datasets,transforms
def train_dre_without_generators(dre_model, projector, dre_path):
    #breakpoint()

    dre_optim = torch.optim.Adam( dre_model.parameters(),lr=1e-2)

    dre_model.train()
    projector.eval()
    

    transform =  transforms.Compose([projector.transform_preprocess, transforms.ToTensor()])
    
    iterations = 10000
    pbar = tqdm(total=iterations, leave=False)
  
    bs = 128
    data_loader1 = torch.utils.data.DataLoader(datasets.ImageFolder('/usr/WS2/olson60/research/StyleGan3/t_imagefolder/', transform=transform), batch_size=bs, shuffle = True, pin_memory=True, num_workers=8,drop_last=True)
    data_loader2 = torch.utils.data.DataLoader(datasets.ImageFolder('/usr/WS2/olson60/research/StyleGan3/r_imagefolder/', transform=transform), batch_size=bs, shuffle = True, pin_memory=True, num_workers=8,drop_last=True)

    outlier_loss_list = []
    i = 0
    while True:
        for ((img1,_),(img2,_)) in zip(data_loader1,data_loader2):
            i+=1
            if i > iterations: 
                torch.save(dre_model.state_dict(), dre_path)
                pbar.close()
                return dre_model 
            dre_optim.zero_grad()
            imgs = torch.cat([img1,img2],dim=0).cuda()

            with torch.no_grad(): feats = projector(imgs)
                
            dre_logit = dre_model(feats)
            
            dre_logit1 = dre_logit[:, 0]
            dre_logit2 = dre_logit[:, 1]

            inlier_loss = -torch.log(dre_logit1[:bs]).mean() - torch.log(dre_logit2[bs:]).mean()
            outlier_loss = dre_logit1[bs:].mean() + dre_logit2[:bs].mean()

            loss = inlier_loss + outlier_loss
            loss.backward()
            dre_optim.step()

            inlier_avg1 = dre_logit1[:bs].mean().item()
            inlier_avg2 = dre_logit2[bs:].mean().item()

            outlier_avg1 = dre_logit1[bs:].mean().item()
            outlier_avg2 = dre_logit2[:bs].mean().item()

            outlier_loss_list.append(outlier_loss.item())
            if (inlier_avg1 + inlier_avg2)/2 > 49 and i > 150: break
            pbar.update()
            pbar.set_postfix_str(f"in1 {inlier_avg1:.3f}, in2 {inlier_avg2:.3f}, o1 {outlier_avg1:.3f}, o2 {outlier_avg2:.3f}, iloss {inlier_loss.item():.3f}, oloss {outlier_loss.item():.3f}", refresh=False)
            torch.cuda.empty_cache()

    pbar.close()
    print("outlir losses:")
    for i in outlier_loss_list: print(i)
    return dre_model    


def train_bce(bce_model, generator, generator2, projector):

    bce_optim = torch.optim.Adam( bce_model.parameters(),lr=1e-2)

    bce_model.train()
    projector.eval()
    generator.eval()
    generator2.eval()

    resize = projector.resize

    with torch.no_grad():
        bs = 4
        try:
            while True: 
                img1 = generator( generator.sample_latent(bs*2))
                img2 = generator2(generator2.sample_latent(bs*2))
                imgs = torch.cat([resize(img1),resize(img2)],dim=0)
                feats = projector(imgs)
                bs*=2
                print("working batchsize: ", bs)
                if bs == 32: break
        except:
            print("optimal batch size found at: ", bs)
    
    iterations = 1000
    pbar = tqdm(total=iterations, leave=False)

    outlier_loss_list = []
    for i in range(iterations):
        bce_optim.zero_grad()
        with torch.no_grad():
            img1 = generator( generator.sample_latent(bs))
            img2 = generator2(generator2.sample_latent(bs))
            imgs = torch.cat([resize(img1),resize(img2)],dim=0)

            feats = projector(imgs)
        #import pdb; pdb.set_trace()
        bce_logit = bce_model(feats)
        bce_logit1 = bce_logit[:, 0]
        bce_logit2 = bce_logit[:, 1]

        bce_prob = torch.sigmoid(bce_logit)
        
        bce_prob1 = bce_prob[:, 0]
        bce_prob2 = bce_prob[:, 1]

        inlier_loss = -torch.log(bce_prob1[:bs]).mean() - torch.log(bce_prob2[bs:]).mean()
        outlier_loss = torch.log(bce_prob1[bs:]).mean() + torch.log(bce_prob2[:bs]).mean()

        loss = inlier_loss + outlier_loss
        loss.backward()
        bce_optim.step()
        #if i % 25 == 0: import pdb; pdb.set_trace()
        with torch.no_grad():
            inlier_avg1 = bce_logit1[:bs].mean().item()
            inlier_avg2 = bce_logit2[bs:].mean().item()

            outlier_avg1 = bce_logit1[bs:].mean().item()
            outlier_avg2 = bce_logit2[:bs].mean().item()

            outlier_loss_list.append(outlier_loss.item())
        if (inlier_avg1 + inlier_avg2)/2 > 49: break
        pbar.update()
        pbar.set_postfix_str(f"in1 {inlier_avg1:.3f}, in2 {inlier_avg2:.3f}, o1 {outlier_avg1:.3f}, o2 {outlier_avg2:.3f}, iloss {inlier_loss.item():.3f}, oloss {outlier_loss.item():.3f}", refresh=False)

    pbar.close()
    print("outlir losses:")
    for i in outlier_loss_list: print(i)
    return bce_model

@torch.no_grad()
def generate(cfg: DictConfig) -> None:
    """Generates images from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)
    cfg.n_dirs = list(range(cfg.k))
    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)

    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    cfg.model['size'] = generator2.n_latent()
    model2: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)

    # Preload model
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model2.load_state_dict(checkpoint["model2"])
    #import pdb; pdb.set_trace()
    '''projector: Projector = instantiate(cfg.projector, img_preprocess='resize', min_resolution=min(generator.resolution, generator2.resolution)).to(device)
    projector.eval()
    f1s = []
    f2s = []
    for i in range(10):
        z1_orig = generator.sample_latent(100).to(device)
        z2_orig = generator2.sample_latent(100).to(device)

        img1 = generator.get_features(z1_orig)
        img2 = generator.get_features(z2_orig)

        img1 = projector.transform_preprocess(img1)
        img2 = projector.transform_preprocess(img2)

        f1 = projector(img1)
        f2 = projector(img2)

        f1s.extend(f1.cpu().numpy())
        f2s.extend(f2.cpu().numpy())

    import numpy as np
    f1s = np.array(f1s)
    f2s = np.array(f2s)
    #import pdb; pdb.set_trace()



    from sklearn import manifold
    X_embedded = manifold.TSNE(n_components=2).fit_transform( np.concatenate([f1s,f2s]))
    from matplotlib import pyplot as plt


    # Set the figure size
    plt.rcParams["figure.figsize"] = [8.00, 8.0]
    plt.rcParams["figure.autolayout"] = True
    yy = ["r"] * 1000 + ["b"] * 1000
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=yy , s= 50)#, cmap='hot')
    #plt.savefig(os.path.join( hydra.utils.get_original_cwd(),"red_nohat_vs_blue_noglass3.png"))
    plt.savefig(os.path.join( hydra.utils.get_original_cwd(),"red_celeba_vs_blue_metface3.png"))
    plt.clf()
    exit()'''

    invert_z_file = os.path.join( hydra.utils.get_original_cwd(), 'invertedpoints',cfg['generator']['class_name']+ "_" +cfg['generator2']['class_name'] + ".pt")
    cfg.n_samples = 16

    #import pdb; pdb.set_trace()
    '''if os.path.exists(invert_z_file):
        zs = torch.load(invert_z_file, map_location=device)
        if type(zs) is dict:
            z1 = zs['z1']
            z2 = zs['z2']
        else:
            z1 = zs[:cfg.n_samples*2,0]
            z2 = zs[:cfg.n_samples*2,1]
    
        if generator.w_primary:  z1 = generator.convert_z2w(z1)
        if generator2.w_primary: z2 = generator2.convert_z2w(z2)

    else:
        z1, z2 = invert_2_gens(generator, generator2, cfg.n_samples)
        save_zs = {'z1':z1, 'z2':z2}
        torch.save(save_zs,invert_z_file)'''

    #import pdb; pdb.set_trace()
    #new_alphas = [a/2 for a in cfg.alphas]
    new_alphas = cfg.alphas
    visualizer = Visualizer(
        model=model,
        generator=generator,
        device=device,
        n_samples=cfg.n_samples,
        #n_samples=z1,
        n_dirs=cfg.n_dirs,
        alphas=new_alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        gen_ind=1,
        #image_size=128,
    )
    #new_alphas = [a/10 for a in cfg.alphas]
    visualizer2 = Visualizer(
        model=model2,
        generator=generator2,
        device=device,
        n_samples=cfg.n_samples,
        #n_samples=z2,
        n_dirs=cfg.n_dirs,
        alphas=new_alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        gen_ind=2,
        #image_size=128,
    )
    clamp_val = 1.0
    visualizer.visualize()
    #visualizer2.visualize()

    visualizer.visualize(clamp_val=clamp_val)
    #visualizer2.visualize(clamp_val=clamp_val)




def get_device(cfg: DictConfig) -> torch.device:
    """Initializes the device from config

    Args:
        cfg: Hydra config

    Returns:
        device on which the model will be trained or evaluated

    """
    if cfg.auto_cpu_if_no_gpu:
        device = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(cfg.device)

    return device




import torchvision.transforms as transforms

def invert_2_gens(G1, G2, n_samples, z_clamp=.3):
    convert2w_1 = G1.w_primary
    convert2w_2 = G2.w_primary
    #G1.model.custom_out_resolution = 128
    #G2.model.custom_out_resolution = 128

    G1.use_z()
    G2.use_z()

    #get 2 semi-random starting point images
    z1 = G1.sample_latent(n_samples).detach() 
    z2 = G2.sample_latent(n_samples).detach() / 10

    z1_bonus = G1.sample_latent(n_samples).detach() 
    z2_bonus = G2.sample_latent(n_samples).detach() / 5

    z1.requires_grad = True
    z2.requires_grad = True


    optimizer1 = torch.optim.Adam([z1], lr=1e-3)
    optimizer2 = torch.optim.Adam([z2], lr=1e-3)

    pbar = tqdm(range(50))
    latent_path = []
    percept = lpips.LPIPS(net='vgg', spatial=True).to(G1.device)

    l2_lam = .1
    mse_lam = 1

    set_resize = True

    for i in pbar:
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        #break
        img1 = G1(z1)
        img2 = G2(z2)

        if set_resize:
            if img1.shape[-1] == img2.shape[-1]: 
                resize = torch.nn.Identity()
            else:
                resize = transforms.Resize(min(img1.shape[-1],img2.shape[-1]))
            set_resize = False

        img1 = resize(img1)
        img2 = resize(img2)

        mse_loss = mse_lam*F.mse_loss(img1, img2)
        l2_loss =torch.zeros(1).cuda() # = l2_lam *((z1 **2 ).mean() + (z2 **2).mean())
        p_loss = 20*percept(img1, img2).mean()

        loss =  mse_loss + l2_loss + p_loss

        loss.backward()
        if i < 20: optimizer1.step()
        optimizer2.step()
        with torch.no_grad():
            #z1.clamp(-z_clamp,z_clamp) 
            z2.clamp(-z_clamp,z_clamp) 

        pbar.set_description( f' mse: {mse_loss.item():.4f}; gaussian: {l2_loss.item():.4f}')
    #
    if convert2w_1: 
        z1 = G1.convert_z2w(z1)
        z1_bonus = G1.convert_z2w(z1_bonus)
        G1.use_w()
    if convert2w_2: 
        z2 = G2.convert_z2w(z2)
        z2_bonus = G2.convert_z2w(z2_bonus)
        G2.use_w()


    print(z1.max())
    print(z2.max())
    print(z1.min())
    print(z2.min())
    #import pdb; pdb.set_trace()
    z1 = torch.cat([z1.detach(), z1_bonus],dim=0)
    z2 = torch.cat([z2.detach(), z2_bonus],dim=0)

    return z1 , z2 #, img1, img2


from torch.cuda.amp import custom_bwd, custom_fwd
class DifferentiableClamp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, max, min):
        if min is None: return input.clamp(max=max)
        if max is None: return input.clamp(min=min)
        return input.clamp(max=max, min = min)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def dclamp(input, max=None, min=None):
    return DifferentiableClamp.apply(input, max, min)

class TransNet(nn.Module):
    def __init__(self, nc=18, mixing=True, affine=True, act='lrelu', 
        clamp=True, num_blocks=4, a=0.5):
        super(TransNet, self).__init__() 
        if act == 'relu':
            self.act = nn.ReLU() 
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act == nn.Sigmoid() 
        # parameters
        self.mixing = mixing 
        self.affine = affine
        self.clamp = clamp
        #conv blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(3, nc, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.2)
            )
        self.block2 = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.2)
            )
        self.block3 = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.2)
            )
        self.block4 = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.2)
            )
        self.block5 = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.2)
            )
        self.final = nn.Sequential(
            nn.Conv2d(nc, 3, kernel_size=3, stride=1, padding=1), 
            )
        blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
        self.blocks = []
        for bb in range(num_blocks):
            self.blocks.append(blocks[bb])

        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.gamma.requires_grad = True

        self.tau = torch.nn.Parameter(torch.ones(1))
        self.tau.requires_grad = True

        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.beta.requires_grad = True
        
    def forward(self, x):
        orig = x
        for bb in range(len(self.blocks)):
            x = self.blocks[bb](x)
            if torch.isnan(x).any(): print("layer {} problem".format(bb))
        out = self.final(x)
        if self.mixing: 
            out = self.tau*orig + (1-self.tau)*out
        if self.affine:
            out = out*self.gamma + self.beta

        out = dclamp(out, min=-1, max=1)
        
        return out