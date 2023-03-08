import logging
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from colat.generators import Generator
from colat.metrics import LossMetric

import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision import transforms


ATTS = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]


class Evaluator:
    """Model evaluator

    Args:
        model: model to be evaluated
        loss_fn: loss function
        generator: pretrained generator
        projector: pretrained projector
        device: device on which to evaluate model
        iterations: number of iterations
    """

    def __init__(
        self,
        model: torch.nn.Module,
        generator: Generator,
        device: torch.device,
        batch_size: int,
        iterations: int,
        att_model_path: str,
        total_directions: int,
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Model
        self.model = model
        self.generator = generator

        # Iterations & batch size
        self.iterations = iterations
        self.batch_size = batch_size
        self.total_directions = total_directions

        # Metrics
        self.att_classifier = AttClsModel().to(device)
        self.att_classifier.load_state_dict(torch.load(att_model_path))

    @torch.no_grad()
    def evaluate(self, truncate_val=None) -> float:
        """Evaluates the model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """


        # Set to eval
        self.generator.eval()
        self.model.eval()

        iters = 4

        # Loop
        def cat(x1,x2): return torch.cat([x1, x2],dim=0)

        #import pdb; pdb.set_trace()
        #breakpoint()
        entropy_by_direction = []
        att_diff_by_direction = []
        diveristy = []
        for k in tqdm(range(self.total_directions)):
            att_diff_list = []
            att_diffs_list = []
            att_prob_list = []

            for i in tqdm(range(iters)):
                z_orig = self.generator.sample_latent(self.batch_size)
                z_d_k = self.model.forward_single(z_orig, k)
                if truncate_val is not None:
                    z_orig = self.generator.truncate_w(z_orig, truncate_val)
                    z_d_k  = self.generator.truncate_w(z_d_k, truncate_val)


                #RETURNS A NORMAL IMAGE
                all_stats = self.att_classifier(self.generator(cat(z_orig, z_d_k)))
                orig_stats, d_stats = torch.split(all_stats, [z_orig.shape[0],z_d_k.shape[0]])
                att_diff = torch.abs(orig_stats - d_stats)
                att_diffs_list.extend(att_diff)
                att_diff_probs = F.softmax(att_diff, dim=1)
                att_diff_entropy = -(att_diff_probs * torch.log(att_diff_probs)).mean(1)
                att_diff_list.extend(att_diff_entropy)
                att_prob_list.extend(att_diff_probs)

            entropy_by_direction.append(torch.stack(att_diff_list).mean())
            att_diff_by_direction.append(torch.stack(att_diffs_list).mean())
            diveristy.append(torch.stack(att_prob_list).mean(0))
        #import pdb; pdb.set_trace()
        #most_changed_direction = torch.argmax(torch.tensor(att_diff_by_direction))
        #with open(f'most_changed_dir_is_{most_changed_direction}.txt', 'w') as f:       f.write(f"{most_changed_direction}")
        #exit()
        #score = torch.stack(entropy_by_direction).mean().item() 
        score = torch.stack(diveristy).std(0).mean().item()
        return score#, torch.stack(entropy_by_direction)




class AttClsModel(nn.Module):
    def __init__(self):
        super(AttClsModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        hidden_size = 2048
        
        #self.lambdas = torch.ones((40,), device=device)
        self.resize = transforms.Resize(256)
        
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
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        return x



    def forward(self, x):
        x = self.resize(x)
        x = (x * 2) -1
        x = self.backbone_forward(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
        