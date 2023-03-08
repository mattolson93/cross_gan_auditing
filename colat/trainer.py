import logging
import os
import time
from typing import List, Optional

import torch
import tqdm
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from colat.generators import Generator
from colat.metrics import LossMetric

import matplotlib.pyplot as plt
import numpy as np
from colat.utils.net_utils import create_dre_model

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
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        generator: Generator,
        projector: torch.nn.Module,
        batch_size: int,
        iterations: int,
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
        dre_path: str = "",
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
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = generator
        self.projector = projector
        self.train_projector = train_projector
        self.feed_layers = feed_layers

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

        self.val_acc_metric = LossMetric()
        self.val_loss_metric = LossMetric()

        # Best
        self.best_loss = -1
        if dre_path != "":
            self.dre_model = create_dre_model(layers=4)
            self.dre_model.load_state_dict(torch.load(dre_path))
        else:
            self.dre_model = None
        self.generator.model.custom_out_resolution = 128

    def dre_loss(self, feats):
        logits = F.softplus(dclamp(self.dre_model(feats),min=-50, max=50))
        loss = -torch.log(logits)
        return loss.mean()  


    def train(self) -> None:
        """Trains the model"""
        self.logger.info("Beginning training")
        start_time = time.time()

        epoch = 0
        iteration = self.start_iteration
        while iteration < self.iterations + 1:
            if iteration + self.eval_freq < self.iterations:
                num_iters = self.eval_freq
            else:
                num_iters = max(self.iterations - iteration, 1)

            start_epoch_time = time.time()
            if self.mixed_precision:
                self._train_loop_amp(epoch, num_iters)
            else:
                do_val = self._train_loop(epoch, num_iters)

            if do_val: self._val_loop(epoch, self.eval_iters)

            epoch_time = time.time() - start_epoch_time
            self._end_loop(epoch, epoch_time, iteration)

            iteration += num_iters
            epoch += 1

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(
            os.path.join(self.save_path, "final_model.pt"), self.iterations
        )

    def _custom_train_loop(self, epoch: int, iterations: int):
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")



        
        if not (self.loss_fn.name == "voynov" or self.loss_fn.name == "hessian" or self.loss_fn.name == "jacobian"):
            raise ValueError(f"loss func must be 'voynov' or 'hessian' or 'simclr' " )


        #if isinstance(self.model.alpha, list) and self.loss_fn.name == "jacobian":
        #    raise ValueError(f"model alphas need to be a 2 length list [-5,5] and not {self.model.alpha}" )
        #elif not isinstance(self.model.alpha, list) and not self.loss_fn.name == "jacobian":
        #    raise ValueError(f"model alphas need a single value (1 or 3) and not {self.model.alpha}" )

        # Set to train
        self.model.train()

        # Set to eval
        self.generator.eval()

        if self.train_projector:
            self.projector.train()
        else:
            self.projector.eval()

        def fast_gram_schmidt(vv):
            def gs(x, ys):
                for y in ys:
                    x = x - x.dot(y) / y.dot(y) * y
                return x

            nk = vv.size(0)
            uu = [vv[0]]
            for k in range(1, nk):
                uu.append(gs(vv[k], uu))
            uu = torch.stack(uu)
            uu = uu.div(uu.pow(2).sum(dim=1, keepdim=True).pow(0.5))
            return uu

        acc = 0.0
        #orthogonalize the 
        with torch.no_grad(): self.model.params.copy_(fast_gram_schmidt(self.model.params))
        self.optimizer.zero_grad()
        #deformator1.linear.weight.data = F.normalize(deformator1.linear.weight.data,p=1, dim=1)

        
        if self.loss_fn.name == "voynov" and not hasattr(self, 'pred_model'):
            self.pred_model = torch.nn.Linear(self.model.size, self.model.k).cuda() 
            self.pred_model_eps = torch.nn.Linear(self.model.size, 1).cuda() 
            self.pred_model_opt = torch.optim.Adam(list(self.pred_model.parameters()) + list(self.pred_model_eps.parameters()),lr=1e-2)
            self.ce_loss = torch.nn.CrossEntropyLoss().cuda()


        for i in range(iterations):

            #get zs
            #sample a few directions randomly
            #epsilon = randomly sample the distances from [-alpha,alpha]

            z = self.generator.sample_latent(self.batch_size).to(self.device)


            

            if self.loss_fn.name == "hessian": 
                z_pos, z_neg = self.model(z, pos_and_neg=True)
                epsilon= self.model.sampled_alphas

                f_og  =  self.projector(self.generator(z)).repeat(self.model.batch_k,1)
                f_pos =  self.projector(self.generator(z_pos))
                f_neg =  self.projector(self.generator(z_neg))


                loss = ((f_pos - (2 * f_og) + f_neg) / (epsilon ** 2)).mean()
                loss.backward()
                #breakpoint()

            if self.loss_fn.name =='jacobian':


                z_pos, _ = self.model(z)

                f_og  =  self.projector(self.generator(z)).repeat(self.model.batch_k,1)
                f_pos =  self.projector(self.generator(z_pos))

                loss = ((f_pos - f_og) / self.model.sampled_alphas).mean()

                loss.backward()

            elif self.loss_fn.name == "voynov":
                updated_z, _ = self.model(z)
                labels = self.model.selected_k.repeat(self.batch_size).cuda()

                orig_imgs = self.generator(z).repeat(self.model.batch_k,1,1,1)
                updated_imgs = self.generator(updated_z)
                pred_feats = self.projector(torch.cat([orig_imgs,updated_imgs], dim=1))

                shift_prediction = self.pred_model_eps(pred_feats)
                logits = self.pred_model(pred_feats)


                #breakpoint()
                logit_loss = self.ce_loss(logits, labels)
                alpha_pred_loss = 0.25 * torch.abs(shift_prediction - self.model.sampled_alphas).mean()
                loss = logit_loss + alpha_pred_loss

                loss.backward()
                self.pred_model_opt.step()
                self.pred_model_opt.zero_grad()






            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            with torch.no_grad(): self.model.params.copy_(fast_gram_schmidt(self.model.params))

            # Update metrics
            self.train_acc_metric.update(acc, z.shape[0])
            self.train_loss_metric.update(loss.item(), z.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc:.3f} Loss: {loss.item():.3f}", refresh=False
            )

        pbar.close()
        return 'voynov' not in self.loss_fn.name



    def _train_loop(self, epoch: int, iterations: int) -> None:
        if self.loss_fn.name != "simclr": return self._custom_train_loop(epoch,iterations)

        """
        Regular train loop

        Args:
            epoch: current epoch
            iterations: iterations to run model
        """
        # Progress bar
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Set to eval
        self.generator.eval()

        if self.train_projector:
            self.projector.train()
        else:
            self.projector.eval()

        for i in range(iterations):
            # To device
            z = self.generator.sample_latent(self.batch_size)
            z = z.to(self.device)
            z_orig = z

            # Original features
            with torch.no_grad():
                orig_feats = self.generator.get_features(z)
                orig_feats = self.projector(orig_feats)

            # Apply Directions
            self.optimizer.zero_grad()
            #import pdb; pdb.set_trace()
            z, z_nograd = self.model(z)

            # Get features
            feats = self.generator.get_features(z)
            feats = self.projector(feats)
            with torch.no_grad():
                feats_no_grad = self.generator.get_features(z_nograd)
                feats_no_grad = self.projector(feats_no_grad)

            dreloss = self.dre_loss(feats) if self.dre_model is not None else 0


            feats =  torch.cat([feats, feats_no_grad],dim=0)
            # Take feature divergence
            feats = feats - orig_feats.repeat(self.model.k,1)
            features = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

            # Loss
            acc, loss = self.loss_fn(features)
            loss = loss + dreloss
            loss.backward()

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            self.train_acc_metric.update(acc.item(), z.shape[0])
            self.train_loss_metric.update(loss.item(), z.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
            )

        pbar.close()
        return True

    def _train_loop_amp(self, epoch: int, iterations: int) -> None:
        """
        Train loop with Automatic Mixed Precision

        Args:
            epoch: current epoch
            iterations: iterations to run model
        """
        # Progress bar
        pbar = tqdm.tqdm(total=len(iterations), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Loop
        for i in range(iterations):
            # To device
            z = self.generator.sample_latent(self.batch_size)
            z = z.to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()

            # Use amp in forward pass
            with autocast():
                # Original features
                with torch.no_grad():
                    orig_feats = self.generator.get_features(z)#
                    orig_feats = self.projector(orig_feats)

                # Apply Directions
                z = self.model(z)

                # Forward
                features = []
                for j in range(z.shape[0] // self.batch_size):
                    # Prepare batch
                    start, end = j * self.batch_size, (j + 1) * self.batch_size

                    # Get features
                    feats = self.generator.get_features(z[start:end, ...])
                    feats = self.projector(feats)

                    # Take feature divergence
                    feats = feats - orig_feats
                    feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                    features.append(feats)
                features = torch.cat(features, dim=0)

                # Loss
                acc, loss = self.loss_fn(features)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            # Update optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            # Update metrics
            self.train_acc_metric.update(acc.item(), z.shape[0])
            self.train_loss_metric.update(loss.item(), z.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss: {loss.item():.3f} ", refresh=False
            )

        pbar.close()

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

        # Loop
        for i in range(iterations):
            with torch.no_grad():
                # To device
                z = self.generator.sample_latent(self.batch_size)
                z = z.to(self.device)
                z_orig = z

                # Original features
                with torch.no_grad():
                    orig_feats = self.generator.get_features(z)
                    orig_feats = self.projector(orig_feats)

                # Apply Directions
                self.optimizer.zero_grad()
                #import pdb; pdb.set_trace()
                z, z_nograd = self.model(z)

                # Get features
                feats = self.generator.get_features(z)
                feats = self.projector(feats)
                with torch.no_grad():
                    feats_no_grad = self.generator.get_features(z_nograd)
                    feats_no_grad = self.projector(feats_no_grad)



                feats =  torch.cat([feats, feats_no_grad],dim=0)
                # Take feature divergence
                feats = feats - orig_feats.repeat(self.model.k,1)
                features = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                # Loss
                acc, loss = self.loss_fn(features)
                self.val_acc_metric.update(acc.item(), z.shape[0])
                self.val_loss_metric.update(loss.item(), z.shape[0])

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(
                    f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
                )

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float, iteration: int):
        # Print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # Write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # Save model
        if self.save_path is not None:
            #self._save_model(os.path.join(self.save_path, "most_recent.pt"), iteration)
            self._save_cosines(self.model , f"cosine.png")

        eval_loss = self.val_loss_metric.compute()
        if self.best_loss == -1 or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            #self._save_model(os.path.join(self.save_path, "best_model.pt"), iteration)
            self._save_cosines(self.model , f"cosine_best.png")

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
        if self.train_projector: self.projector.load_state_dict(checkpoint["projector"])
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