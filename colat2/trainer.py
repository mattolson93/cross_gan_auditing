import logging
import os
import time
from typing import List, Optional

import torch

import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from colat.generators import Generator
from colat.metrics import LossMetric


import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import torch
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

        #self.optimizer.param_groups.append({'params': self.dre_model.parameters() })
        self.softplus = torch.nn.Softplus()

    #How this should really work:
    #an image generated from dataset 1 unique is a test instance where overlap from either
    #is a train instance
    #2 DRE models where each treats its own gen's unique as the test set
    def dre_update(self, feats1, feats2):
        self.dre_optim.zero_grad()
        dre_logit = self.dre_model(torch.cat([feats1.detach(), feats2.detach()], dim=0))
        dre_logit = self.softplus(dclamp(dre_logit,min=-50, max=50))

        f1s = feats1.shape[0]

        dre_logit1 = dre_logit[:, 0]
        dre_logit2 = dre_logit[:, 1]

        inlier_loss = -torch.log(dre_logit1[:f1s]) - torch.log(dre_logit2[f1s:])
        outlier_loss = dre_logit1[f1s:] + dre_logit2[:f1s]


        loss = inlier_loss.mean() + outlier_loss.mean()
        loss.backward()
        self.dre_optim.step()
        self.dre_optim.zero_grad()
        return loss.item()


    def dre_loss_fn(self, feats1, feats2, overlap_inds, bs):
        dre_logit = self.dre_model(torch.cat([feats1, feats2], dim=0))
        dre_logit = self.softplus(dclamp(dre_logit,min=-50, max=50))

        f1s = feats1.shape[0]

        dre_logit1 = dre_logit[:, 0]
        dre_logit2 = dre_logit[:, 1]
        

        #unique indices should maximize the logit
        #overlap should minimize the logit
        unique_inds1 = torch.cat([(~overlap_inds).repeat_interleave(bs), torch.zeros(len(feats2)).bool()])
        unique_inds2 = torch.cat([torch.zeros(len(feats1)).bool(), (~overlap_inds).repeat_interleave(bs)])
        overlap_inds = (overlap_inds).repeat_interleave(bs).repeat(2)

        only_overlap = overlap_inds.sum() == overlap_inds.shape[0]
        only_unique  = overlap_inds.sum() == 0

        inlier_loss  = 0 if only_unique  else -torch.log(dre_logit[overlap_inds]).mean() 
        outlier_loss = 0 if only_overlap else dre_logit1[unique_inds1].mean() + dre_logit2[unique_inds2].mean()

        #print(inlier_loss, outlier_loss)
        #
        return inlier_loss + outlier_loss


    def train(self) -> None:
        """Trains the model"""
        self.logger.info("Beginning training")
        start_time = time.time()

        epoch = 0
        iteration = self.start_iteration
        self._val_loop(epoch, 0)

        while iteration < self.iterations:
            if iteration + self.eval_freq < self.iterations:
                num_iters = self.eval_freq
            else:
                num_iters = self.iterations - iteration

            start_epoch_time = time.time()
            if self.mixed_precision:
                self._train_loop_amp(epoch, num_iters)
            else:
                self._train_loop(epoch, num_iters)

            #self._val_loop(epoch, self.eval_iters)

            epoch_time = time.time() - start_epoch_time
            #self._end_loop(epoch, epoch_time, iteration)

            iteration += num_iters
            epoch += 1

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(
            os.path.join(self.save_path, "final_model.pt"), self.iterations
        )

    def forward_projector(self,x, which_cnn=None):
        x1, x2 = x
        if self.do_multi_cnn:
            ret1 = self.projector(x1, 1)
            ret2 = self.projector(x2, 2)
        else:
            x1 = self.projector.net.resize(x1)
            x2 = self.projector.net.resize(x2)
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

        #import pdb; pdb.set_trace()
        for i in range(iterations):
            # To device
            #z = self.wrapped_generator("sample_latent",self.batch_size)
            with torch.no_grad():
                z1_orig = self.generator.sample_latent(self.batch_size).to(self.device)
                z2_orig = self.generator2.sample_latent(self.batch_size).to(self.device)

            # Apply Directions
            self.optimizer.zero_grad()

            z1_grad, z1_nograd = self.model(z1_orig)
            z2_grad, z2_nograd = self.model2(z2_orig,self.model.selected_k, self.model.unselected_k)


            # Original features
            with torch.no_grad():
                orig_feats1_combo = self.generator.get_features(cat(z1_orig, z1_nograd))
                orig_feats2_combo = self.generator2.get_features(cat(z2_orig, z2_nograd))
                #big gan hack
                #orig_feats2_combo = self.generator2.get_features(cat(z2_orig, z2_nograd))


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
            #the first two elements of features1 (for example) are from the SAME direction, just different starting zs
            overlaps = torch.cat([self.model.selected_k,self.model.unselected_k]) < self.overlap_k
            f1 = cat(features1,feats1_nograd)
            f2 = cat(features2,features2_nograd)
            acc, loss_simclr = self.loss_fn(f1,f2, overlaps , self.batch_size)
            
            
            if self.dre_lamb > 0:
                #import pdb; pdb.set_trace()
                dre_internal_loss = self.dre_update(cat(features1,feats1_nograd),cat(features2,features2_nograd))
                loss_dre = self.dre_loss_fn(features1, features2, self.model.selected_k< self.overlap_k, self.batch_size)
            else:
                loss_dre = torch.zeros(1).cuda()

            loss = loss_simclr + (self.dre_lamb * loss_dre )


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
            self.train_acc_metric.update(acc.item(), features1.shape[0])
            self.train_loss_metric.update(loss.item(), features1.shape[0])

            '''total_norm=0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            total_norm2=0
            for p in self.model2.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm2 += param_norm.item() ** 2
            total_norm2 = total_norm2 ** (1. / 2)

            #self.train_grad_metric1.update(self.model.params.grad.mean().item() , z1.shape[0])
            #self.train_grad_metric2.update(self.model2.params.grad.mean().item(), z1.shape[0])
            if i % 10 == 0:
                print("grad1 norm", total_norm)
                print("grad2 norm", total_norm2)
            '''
            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc{acc.item():.3f} Loss{loss_simclr.item():.3f} ldre: {loss_dre.item():.2f}", refresh=False
            )

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
        #    self._save_model(os.path.join(self.save_path, "most_recent.pt"), iteration)

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
            "projector": self.projector.state_dict(),
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


'''
from torchvision import models
import torch
import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = DataParallelPassthrough(model)

    def forward(self, func_name, *inputs):
        class_method = getattr(self.model, func_name)
        return class_method(*inputs)

#B = DistributedDataParallel(Wrapper(), ...)

class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class AttClsModel(nn.Module):
    def __init__(self, model_type):
        super(AttClsModel, self).__init__()
        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            hidden_size = 2048
        elif model_type == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            hidden_size = 512
        elif model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            hidden_size = 512
        else:
            raise NotImplementedError
        #self.lambdas = torch.ones((40,), device=device)
        self.val_loss = []  # max_len == 2*k
        self.fc = nn.Linear(hidden_size, 40)
        self.dropout = nn.Dropout(0.5)

    def backbone_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        #x = self.backbone.layer4(x)

        #x = self.backbone.avgpool(x)

        return x

    def forward(self, input, labels=None):
        x = self.backbone_forward(input)
        x = torch.flatten(x, 1)
        #x = self.dropout(x)
        #x = self.fc(x)

        return x
        
'''