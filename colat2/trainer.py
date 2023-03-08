import logging
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import random 

import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from colat.generators import Generator
from colat.metrics import LossMetric
from colat.loss import ContrastiveLoss


import matplotlib.pyplot as plt
#
import numpy as np

from torch.cuda.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

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
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, max, min)

class Trainer:
    """Model trainer

    Args:
        model: model to train
        loss_fn: loss function
        optimizer: model optimizer
        generator: pretrained generator
        projector: pretrained projector
        device: device to train the model on
        batch_size: number of batch elements
        iterations: number of iterations
        scheduler: learning rate scheduler
        grad_clip_max_norm: gradient clipping max norm (disabled if None)
        writer: writer which logs metrics to TensorBoard (disabled if None)
        save_path: folder in which to save models (disabled if None)
        checkpoint_path: path to model checkpoint, to resume training
        mixed_precision: enable mixed precision training

    """

    def __init__(
        self,
        model: torch.nn.Module,
        model2: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        generator: Generator,
        projector: torch.nn.Module,
        batch_size: int,
        iterations: int,
        overlap_k: int,
        device: torch.device,
        visualizer,
        visualizer2,
        eval_freq: int = 1000,
        eval_iters: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip_max_norm: Optional[float] = None,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        mixed_precision: bool = False,
        train_projector: bool = True,
        feed_layers: Optional[List[int]] = None,
        generator2: Generator = None,
        dre_model: torch.nn.Module = None,
        dre_lamb: int =0,
        global_resolution: int = None,
        
        dre_optim : torch.optim.Optimizer = None,
        extra_stuff = {},
        trans_model = None,

    ) -> None:

        # Logging
        self.logger = logging.getLogger()
        self.writer = writer

        # Saving
        self.save_path = save_path

        # Device
        self.device = device

        # Model
        self.model = model
        self.model2 = model2
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = generator
        self.generator2 = generator2
        self.projector = projector
        self.train_projector = train_projector
        self.feed_layers = feed_layers
        self.overlap_k = overlap_k

        #self.generator.model.set_global_resolution(global_resolution)
        #self.generator2.model.set_global_resolution(global_resolution)
        #self.wrapped_generator = Wrapper(generator)
        #self.wrapped_projector = Wrapper(projector)
        #self.wrapped_model     = Wrapper(model)

        #  Eval
        self.eval_freq = eval_freq
        self.eval_iters = eval_iters

        # Scheduler
        self.scheduler = scheduler
        self.grad_clip_max_norm = grad_clip_max_norm

        # Batch & Iteration
        self.batch_size = batch_size
        self.iterations = iterations
        self.start_iteration = 0
        self.cur_iter = 0

        # Floating-point precision
        self.mixed_precision = (
            True if self.device.type == "cuda" and mixed_precision else False
        )
        self.scaler = GradScaler() if self.mixed_precision else None

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.train_acc_metric = LossMetric()
        self.train_loss_metric = LossMetric()
        self.train_grad_metric1 = LossMetric()
        self.train_grad_metric2 = LossMetric()

        self.val_acc_metric = LossMetric()
        self.val_loss_metric = LossMetric()

        #self.att_classifier = AttClsModel("resnet18").cuda().eval()
        #self.att_classifier.load_state_dict(torch.load('/usr/WS2/olson60/research/latentclr/att_classifier.pt'))
        self.multi_gpu = True
        # Best
        self.best_loss = -1

        self.do_multi_cnn =  hasattr(self.projector, 'multiconv') 


        if not hasattr(self.projector, 'layers'): exit("hacked for dre + resnet ONLY")
        self.dre_lamb = dre_lamb
        self.dre_model = dre_model
        self.dre_optim = dre_optim

        self.visualizer = visualizer
        self.visualizer2 = visualizer2

        self.softplus = torch.nn.Softplus()
        self.mse = torch.nn.MSELoss()

        self.exps = extra_stuff
        if self.exp("do_trans") > 0:
            self.trans = trans_model

        #self.optimizer.param_groups.append({'params': self.dre_model.parameters() })
        

        if self.exp("selfclr_lamb") > 0:
            self.generator.save_intermediate()#self.exp("selfclr_layer"))
            self.selfclr_loss = ContrastiveLoss(self.model.batch_k, .5, False, "mean", 0.0)

    def exp(self, key):
        if key in self.exps.keys():
            return self.exps[key]
        else:
            return 0



    def dre_loss_fn(self, feats1, feats2, overlap_inds, bs):
        #import pdb; pdb.set_trace()
        #unique indices should maximize the logit
        #overlap should minimize the logit
        unique_count = (~overlap_inds).sum() * bs
        if unique_count == 0: return (torch.zeros(1) - .001 )[0].cuda()

        #unique_feats1_inds = torch.cat([(~overlap_inds).repeat_interleave(bs), torch.zeros(len(feats2)).bool()])
        #unique_feats2_inds = torch.cat([torch.zeros(len(feats1)).bool(), (~overlap_inds).repeat_interleave(bs)])
        unique_inds = (~overlap_inds).repeat_interleave(bs)

        unique_feats = torch.cat([feats1[unique_inds], feats2[unique_inds]], dim=0)

        dre_logit = self.dre_model(unique_feats)

        #using the DRE that calls dist 2 outliers, make sure these unique directions are small
        l2 = dre_logit[unique_count:, 0] #dist1 is inlier, dist2 is outlier

        #using the DRE that calls dist 1 outliers, make sure these unique directions are small
        l1 = dre_logit[:unique_count, 1]


        #l1 = -torch.log(1-pred1[unique_inds1]) * (pred1[unique_inds1] > .1)
        #l2 = -torch.log(1-pred2[unique_inds2]) * (pred2[unique_inds2] > .1)
        #outlier_loss = 0 if only_overlap else l1.mean() + l2.mean()

        #print(inlier_loss, outlier_loss)
        #
        return l1.mean() + l2.mean()

    def bce_loss_fn(self, feats1, feats2, overlap_inds, bs):
        #import pdb; pdb.set_trace()
        #unique indices should maximize the logit
        #overlap should minimize the logit
        unique_count = (~overlap_inds).sum() * bs
        if unique_count == 0: return (torch.zeros(1) - .001 )[0].cuda()

        unique_inds = (~overlap_inds).repeat_interleave(bs)

        unique_feats = torch.cat([feats1[unique_inds], feats2[unique_inds]], dim=0)

        dre_logit = torch.sigmoid(self.dre_model(unique_feats))

        #using the DRE that calls dist 2 outliers, make sure these unique directions are small
        l2 = dre_logit[unique_count:, 0] #dist1 is inlier, dist2 is outlier

        #using the DRE that calls dist 1 outliers, make sure these unique directions are small
        l1 = dre_logit[:unique_count, 1]


        #l1 = -torch.log(1-pred1[unique_inds1]) * (pred1[unique_inds1] > .1)
        #l2 = -torch.log(1-pred2[unique_inds2]) * (pred2[unique_inds2] > .1)
        #outlier_loss = 0 if only_overlap else l1.mean() + l2.mean()

        #print(inlier_loss, outlier_loss)
        #
        return l1.mean() + l2.mean()


   

    def train(self) -> None:
        """Trains the model"""
        self.logger.info("Beginning training")
        start_time = time.time()

        self.epoch = 0
        iteration = self.start_iteration
        self._val_loop(self.epoch , 0)

        while iteration < self.iterations:
            if iteration + self.eval_freq < self.iterations:
                num_iters = self.eval_freq
            else:
                num_iters = self.iterations - iteration

            start_epoch_time = time.time()
            if self.mixed_precision:
                self._train_loop_amp(self.epoch , num_iters)
            else:
                self._train_loop(self.epoch , num_iters)

            #self._val_loop(self.epoch , self.eval_iters)

            epoch_time = time.time() - start_epoch_time
            #self._end_loop(self.epoch , epoch_time, iteration)

            iteration += num_iters
            self.epoch  += 1
            self._save_model( os.path.join(self.save_path, "final_model.pt"), self.epoch)

            #final visual
            if iteration % 3 == 0:
                self.visualizer.visualize(trainepoch=str(self.epoch))
                self.visualizer2.visualize(trainepoch=str(self.epoch))

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model( os.path.join(self.save_path, "final_model.pt"), self.iterations)

    def forward_projector(self,x, which_cnn=None):
        x1, x2 = x
        if self.do_multi_cnn:
            ret1 = self.projector(x1, 1)
            ret2 = self.projector(x2, 2)
        else:
            x1 = self.projector.transform_preprocess(x1)
            x2 = self.projector.transform_preprocess(x2)
            ret = self.projector(torch.cat([x1,x2],dim=0))
            ret1, ret2 = torch.split(ret, [x1.shape[0],x2.shape[0]])

        return ret1, ret2

    def diff_norm_feats(self, cur_feats, orig_feats, repeat_count):
        cur_feats = cur_feats - orig_feats.repeat(repeat_count,1)
        ret = cur_feats / torch.reshape(torch.norm(cur_feats, dim=1), (-1, 1))
        return ret

    def _train_loop(self, epoch: int, iterations: int) -> None:
        """
        Regular train loop

        Args:
            epoch: current epoch
            iterations: iterations to run model
        """
        # Progress bar
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        
        self.model.train()
        self.model2.train()

        self.generator.eval()
        self.generator2.eval()
        

        if self.train_projector:
            self.projector.train()
        else:
            self.projector.eval()

        def cat(x1,x2): return torch.cat([x1, x2],dim=0)


        gen1_kw = {"do_checkpoint":True} if "StyleGAN3" in self.generator.name else {}
        gen2_kw = {"do_checkpoint":True} if "StyleGAN3" in self.generator2.name else {}

        alternate_grads = "StyleGAN3" in self.generator.name and "StyleGAN3" in self.generator2.name


        for i in range(iterations):
            if alternate_grads:
                model1 = i % 2 == 0
                model2 = not model1

                gen1_kw = {}#{"do_checkpoint":True} if model1 else {}
                gen2_kw = {}#{"do_checkpoint":True} if model2 else {}

                for p in self.model.parameters(): p.requires_grad_(model1)
                for p in self.generator.parameters(): p.requires_grad_(model1)

                for p in self.model2.parameters(): p.requires_grad_(model2)
                for p in self.generator2.parameters(): p.requires_grad_(model2)


            # To device
            #z = self.wrapped_generator("sample_latent",self.batch_size)
            with torch.no_grad():
                z1_orig = self.generator.sample_latent(self.batch_size).to(self.device)
                z2_orig = self.generator2.sample_latent(self.batch_size).to(self.device)

            # Apply Directions
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            #import pdb; pdb.set_trace()
            d1_grad, d1_nograd = self.model(z1_orig)
            d2_grad, d2_nograd = self.model2(z2_orig,self.model.selected_k, self.model.unselected_k)


            # Original features
            with torch.no_grad():
                gen_feats1_combo_nograd = self.generator.get_features( cat(z1_orig, d1_nograd), **gen1_kw)
                torch.cuda.empty_cache()
                if self.exp("selfclr_lamb") > 0: selfclr_feats_orig, _ = torch.split(self.generator.get_intermediate(), [z1_orig.shape[0],d1_nograd.shape[0]])

                gen_feats2_combo_nograd = self.generator2.get_features(cat(z2_orig, d2_nograd), **gen2_kw)
                torch.cuda.empty_cache()
                if self.exp("do_trans") > 0: gen_feats2_combo_nograd = self.trans(gen_feats2_combo_nograd)
                

                orig_gen_feats1, _ = torch.split(gen_feats1_combo_nograd, [z1_orig.shape[0],d1_nograd.shape[0]])
                orig_gen_feats2, _ = torch.split(gen_feats2_combo_nograd, [z2_orig.shape[0],d2_nograd.shape[0]])

                #big gan hack
                #orig_feats2_combo = self.generator2.get_features(cat(z2_orig, z2_nograd))

                orig_d_feats1_combo, orig_d_feats2_combo = self.forward_projector((gen_feats1_combo_nograd, gen_feats2_combo_nograd))

                orig_feats1, feats1_nograd = torch.split(orig_d_feats1_combo, [z1_orig.shape[0],d1_nograd.shape[0]])
                orig_feats2, feats2_nograd = torch.split(orig_d_feats2_combo, [z2_orig.shape[0],d2_nograd.shape[0]])

                diffed_features1_nograd = self.diff_norm_feats(feats1_nograd, orig_feats1, self.model.k - self.model.batch_k)
                diffed_features2_nograd = self.diff_norm_feats(feats1_nograd, orig_feats2, self.model.k - self.model.batch_k)

            
            if alternate_grads:
                gen_feats2 = self.generator2.get_features(d2_grad, **gen2_kw)
                gen_feats1 = self.generator.get_features(d1_grad, **gen1_kw)
            else:
                gen_feats1 = self.generator.get_features(d1_grad, **gen1_kw)
                gen_feats2 = self.generator2.get_features(d2_grad, **gen2_kw)


            if self.exp("do_trans") > 0: gen_feats2 = self.trans(gen_feats2)

            # Forward
            feats1, feats2 = self.forward_projector((gen_feats1, gen_feats2))
            diffed_features1 = self.diff_norm_feats(feats1, orig_feats1, self.model.batch_k)
            diffed_features2 = self.diff_norm_feats(feats2, orig_feats2, self.model.batch_k)

            
            # Loss
            #the first two elements of features1 (for example) are from the SAME direction, just different starting zs
            label_order= torch.cat([self.model.selected_k,self.model.unselected_k])
            f1 = cat(diffed_features1,diffed_features1_nograd)
            f2 = cat(diffed_features2,diffed_features2_nograd)
            acc, loss_simclr = self.loss_fn(f1,f2, label_order < self.overlap_k, self.batch_size, label_order=label_order)
            loss = loss_simclr 
            
            out_string = f"Acc{acc.item():.3f} loss{loss_simclr.item():.3f}"




            if self.dre_lamb > 0:
                #import pdb; pdb.set_trace()
                loss_dre = self.dre_lamb * self.dre_loss_fn(feats1, feats2, self.model.selected_k< self.overlap_k, self.batch_size)
                out_string += f" ldre: {loss_dre.item():.3f}"
                loss += loss_dre 

            if self.dre_lamb < 0:
                loss_bce = -self.dre_lamb * self.bce_loss_fn(feats1, feats2, self.model.selected_k< self.overlap_k, self.batch_size)
                out_string += f" lbce: {loss_bce.item():.3f}"
                loss += loss_bce 


            if self.exp("recon") > 0:
                color_loss1 = self.mse(orig_gen_feats1.repeat(self.model.batch_k,1,1,1) , gen_feats1)
                color_loss2 = self.mse(orig_gen_feats2.repeat(self.model.batch_k,1,1,1) , gen_feats2)
                color_loss = self.exp("recon") * (color_loss1 + color_loss2)
                out_string += f" recon: {color_loss.item():.3f}"
                loss += color_loss 
           
            if self.exp("selfclr_lamb") > 0:
                #import pdb; pdb.set_trace()
                selfclr_feat1 = self.generator.get_intermediate()
                selfclr_feat1 = torch.reshape(selfclr_feat1 - selfclr_feats_orig.repeat(self.model.batch_k,1,1,1),(selfclr_feat1.shape[0],-1))
                selfclr_feat1 = selfclr_feat1 / torch.reshape(torch.norm(selfclr_feat1, dim=1), (-1, 1))
                #if len(selfclr_feat1.shape) == 4 and selfclr_feat1.shape[-1] >=8:
                #    selfclr_feat1 = F.max_pool2d(selfclr_feat1,kernel_size=4) 


                selfclr_acc, selfclr_loss =  self.selfclr_loss(selfclr_feat1)
                selfclr_loss *= self.exp("selfclr_lamb")

                out_string += f" slfclr {selfclr_acc.item():.3f}:{selfclr_loss.item():.3f}"
                loss += selfclr_loss 

            if self.exp("l2latent_lamb") > 0:
                #import pdb; pdb.set_trace()
                l2_latent_loss1 = self.mse(orig_feats1.repeat(self.model.batch_k,1), feats1)
                l2_latent_loss2 = self.mse(orig_feats2.repeat(self.model.batch_k,1), feats2)
                l2_latent_loss = self.exp("l2latent_lamb") * (l2_latent_loss1 + l2_latent_loss2)

                out_string += f" l2latent: {l2_latent_loss.item():.3f}"
                loss += l2_latent_loss 


            self.writer.add_scalar("cur_acc", acc.item(), self.cur_iter)
            self.writer.add_scalar("cur_loss", loss.item(), self.cur_iter)
            self.cur_iter+=1

            #acc = acc1+acc2+acc3
            #loss = loss_overlap + loss_unique1 + loss_unique2
            #
            loss.backward()

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()
            self.scheduler.step()


            # Update metrics
            self.train_acc_metric.update(acc.item(), f1.shape[0])
            self.train_loss_metric.update(loss.item(), f1.shape[0])

            torch.cuda.empty_cache()
            pbar.update()
            pbar.set_postfix_str(out_string, refresh=False)

        pbar.close()

    @torch.no_grad()
    def _val_loop(self, epoch: int, iterations: int) -> None:
        """
        Standard validation loop

        Args:
            epoch: current epoch
            iterations: iterations to run model
        """
        # Progress bar
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # Set to eval
        self.model.eval()
        self.generator.eval()
        self.projector.eval()
        def cat(x1,x2): return torch.cat([x1, x2],dim=0)

        #import pdb; pdb.set_trace()
        for i in range(iterations):
            # To device
            #z = self.wrapped_generator("sample_latent",self.batch_size)
            z1_orig = self.generator.sample_latent(self.batch_size).to(self.device)
            z2_orig = self.generator2.sample_latent(self.batch_size).to(self.device)


            z1_grad, z1_nograd = self.model(z1_orig)
            z2_grad, z2_nograd = self.model2(z2_orig,self.model.selected_k, self.model.unselected_k)


            # Original features
            orig_feats1_combo = self.generator.get_features(cat(z1_orig, z1_nograd))
            orig_feats2_combo = self.generator2.get_features(cat(z2_orig, z2_nograd))


            orig_feats1_combo, orig_feats2_combo = self.forward_projector((orig_feats1_combo, orig_feats2_combo ))

            orig_feats1, feats1_nograd = torch.split(orig_feats1_combo, [z1_orig.shape[0],z1_nograd.shape[0]])
            orig_feats2, feats2_nograd = torch.split(orig_feats2_combo, [z2_orig.shape[0],z2_nograd.shape[0]])

            features1_nograd = self.diff_norm_feats(feats1_nograd, orig_feats1, self.model.k - self.model.batch_k)
            features2_nograd = self.diff_norm_feats(feats2_nograd, orig_feats2, self.model.k - self.model.batch_k)

                
            
            gen_feats1 = self.generator.get_features(z1_grad)
            gen_feats2 = self.generator2.get_features(z2_grad)

            # Forward
            feats1, feats2 = self.forward_projector((gen_feats1, gen_feats2))
            features1 = self.diff_norm_feats(feats1, orig_feats1, self.model.batch_k)
            features2 = self.diff_norm_feats(feats2, orig_feats2, self.model.batch_k)



            #import pdb; pdb.set_trace()
            # Loss
            overlaps = torch.cat([self.model.selected_k,self.model.unselected_k]) < self.overlap_k
            #the first two elements of features1 (for example) are from the SAME direction, just different starting zs
            acc, loss_simclr = self.loss_fn(cat(features1,feats1_nograd),cat(features2,features2_nograd), overlaps , self.batch_size)
            
            self.val_acc_metric.update(acc.item(), features1.shape[0])
            self.val_loss_metric.update(loss_simclr.item(), features1.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss: {loss_simclr.item():.3f}", refresh=False
            )

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float, iteration: int):
        # Print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # Write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # Save model
        #if self.save_path is not None:
        self._save_model(os.path.join(self.save_path, "final_model.pt"), iteration)

        #self._save_cosines(self.model , f"cosine_model1_{iteration}.pt")
        self._save_cosines(self.model , f"cosine_model1.png")
        self._save_cosines(self.model2, f"cosine_model2.png")


        eval_loss = self.val_loss_metric.compute()
        if self.best_loss == -1 or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            #self._save_model(os.path.join(self.save_path, "best_model.pt"), iteration)
            self._save_cosines(self.model , f"cosine_best_model1.png")
            self._save_cosines(self.model2, f"cosine_best_model2.png")

        # Clear metrics
        self.train_loss_metric.reset()
        self.train_acc_metric.reset()
        self.val_loss_metric.reset()
        self.val_acc_metric.reset()

    def _save_cosines(self, model, filename):
        from sklearn.metrics.pairwise import cosine_similarity
        params = model.get_params()
        if params is None: return
        plt.matshow(cosine_similarity(params) - np.identity(model.k))
        plt.colorbar()
        plt.savefig(os.path.join(self.save_path, filename))
        plt.clf()

    def _epoch_str(self, epoch: int, epoch_time: float):
        s = f"Epoch {epoch} "
        s += f"| Train acc: {self.train_acc_metric.compute():.3f} "
        s += f"| Train loss: {self.train_loss_metric.compute():.3f} "
        s += f"| Val acc: {self.val_acc_metric.compute():.3f} "
        s += f"| Val loss: {self.val_loss_metric.compute():.3f} "
        s += f"| Epoch time: {epoch_time:.1f}s"

        return s

    def _write_to_tb(self, iteration):
        self.writer.add_scalar(
            "Loss/train", self.train_loss_metric.compute(), iteration
        )
        self.writer.add_scalar("Acc/train", self.train_acc_metric.compute(), iteration)
        self.writer.add_scalar("Loss/val", self.val_loss_metric.compute(), iteration)
        self.writer.add_scalar("Acc/val", self.val_acc_metric.compute(), iteration)

    def _save_model(self, path, iteration):
        obj = {
            "iteration": iteration + 1,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "model2": self.model2.state_dict() ,
            "projector": self.projector.state_dict() if self.train_projector else None,
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler": self.scaler.state_dict() if self.mixed_precision else None,
        }
        torch.save(obj, os.path.join(self.save_path, path))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model2.load_state_dict(checkpoint["model2"])
        self.projector.load_state_dict(checkpoint["projector"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_iteration = checkpoint["iteration"]

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.mixed_precision and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scheduler"])

        if self.start_iteration > self.iterations:
            raise ValueError("Starting iteration is larger than total iterations")

        self.logger.info(
            f"Checkpoint loaded, resuming from iteration {self.start_iteration}"
        )


