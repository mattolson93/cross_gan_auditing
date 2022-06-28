import logging
from typing import List, Optional

import torch
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
        model2: torch.nn.Module,
        generator: Generator,
        generator2: Generator,
        device: torch.device,
        batch_size: int,
        iterations: int,
        num_unique_directions: int,
        total_directions: int,
        att_model_path: str,
        w_min_max = None,
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Model
        self.model = model
        self.model2 = model2
        self.generator = generator
        self.generator2 = generator2

        # Iterations & batch size
        self.iterations = iterations
        self.batch_size = batch_size 

        self.num_unique_directions = num_unique_directions
        self.total_directions = total_directions
        self.num_overlap_directions = total_directions - num_unique_directions

        # Metrics
        #self.loss_metric = LossMetric()
        self.query_dict = {
            "celeba" :  [] ,
            "female" :  [ATTS.index("Male")] ,
            "male" :  [ATTS.index("Male")] ,
            "nohat" :  [ATTS.index("Wearing_Hat")] ,
            "noglass" :  [ATTS.index("Eyeglasses")] ,
            "nobeard" :  [ATTS.index("No_Beard")] ,
            "nobeardnohat" :  [ATTS.index("No_Beard"),ATTS.index("Wearing_Hat")] ,
            "noglassnosmilenotie" :  [ATTS.index("Eyeglasses"),ATTS.index("Smiling"),ATTS.index("Wearing_Necktie")] ,
        }


        self.att_classifier = AttClsModel().to(device)
        self.att_classifier.load_state_dict(torch.load(att_model_path))

    @torch.no_grad()
    def get_unique_attributes(self, gen, model, unique_directions):
        def cat(x1,x2): return torch.cat([x1, x2],dim=0)

        #import pdb; pdb.set_trace()
        att_diff_dict = []
        for k in tqdm(unique_directions):
            att_diff = []
            for i in tqdm(range(self.iterations)):
                z_orig = gen.sample_latent(self.batch_size)
                z_d_k = model.forward_single(z_orig, k)

                
                all_stats = self.att_classifier(gen(cat(z_orig, z_d_k)))
                orig_stats, d_stats = torch.split(all_stats, [z_orig.shape[0],z_d_k.shape[0]])
                att_diff.extend(orig_stats - d_stats)

            att_diff = torch.stack(att_diff)
            att_diff_dict.append(att_diff.cpu())

        return att_diff_dict


    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluates the model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """

        # Progress bar
        #pbar.set_description("Evaluating... ")
        def cat(x1,x2): return torch.cat([x1, x2],dim=0)

        #import pdb; pdb.set_trace()
        # Set to eval
        self.generator.eval()
        self.generator2.eval()
        self.model.eval()
        self.model2.eval()

        self.batch_size = int(self.batch_size)
        self.iterations = int(self.iterations / 10000) + 1

        overlap_direction_scores = []

        for k in tqdm(range(self.num_overlap_directions)):
            #if k > 1: break
            att_diff_dist1 = []
            att_diff_dist2 = []
            # Loop
            for i in tqdm(range(self.iterations)):
                #load some original images from the generators
                z1_orig = self.generator.sample_latent(self.batch_size).to(self.device)
                z2_orig = self.generator2.sample_latent(self.batch_size).to(self.device)

                z1_d_k = self.model.forward_single(z1_orig, k)
                z2_d_k = self.model2.forward_single(z2_orig, k)

                #load the given direction images
                xs_1 = self.generator(cat(z1_orig, z1_d_k))
                xs_2 = self.generator2(cat(z2_orig, z2_d_k))

                #get all the stats
                all_stats = self.att_classifier(cat(xs_1,xs_2))

                #load to variables correctly
                raw_stats1, raw_stats2 = torch.split(all_stats, [xs_1.shape[0],xs_2.shape[0]])
                att_orig1, att_d_dist1 = torch.split(raw_stats1, [z1_orig.shape[0],z1_d_k.shape[0]])
                att_orig2, att_d_dist2 = torch.split(raw_stats2, [z2_orig.shape[0],z2_d_k.shape[0]])

                att_diff_dist1.extend(att_orig1 - att_d_dist1)
                att_diff_dist2.extend(att_orig2 - att_d_dist2)
            
            #import pdb; pdb.set_trace()
            att_diff_dist1 = torch.stack(att_diff_dist1)
            att_diff_dist2 = torch.stack(att_diff_dist2)

            #average the stats array

            cur_score = ((att_diff_dist1.mean(0) - att_diff_dist2.mean(0)) **2).mean()
            overlap_direction_scores.append(cur_score)

            #disentangled_score1_d1 = entropy(att_diff_d1_dist1)

        unique_directions = range(self.num_overlap_directions, self.total_directions)
        att_diff_uniques1 = self.get_unique_attributes(self.generator, self.model, unique_directions)
        att_diff_uniques2 = self.get_unique_attributes(self.generator2, self.model2, unique_directions)

        #query_dict = self.query_dict 
        #import pdb; pdb.set_trace()

        g1_unique_queries = self.query_dict[self.generator2.outclass]
        n1_queries = len(g1_unique_queries)
        unique_score1 = 0.0
        for query in g1_unique_queries:
            best_rank = 0
            for direction in att_diff_uniques1:
                rankings = torch.sort(torch.abs(direction.mean(0)), descending=True)[1]
                cur_rank = 1/(rankings.tolist().index(query) + 1)
                best_rank = max(cur_rank, best_rank)

            unique_score1 += best_rank

        unique_score1 = unique_score1/n1_queries if n1_queries > 0 else 0.0



        g2_unique_queries = self.query_dict[self.generator.outclass]
        n2_queries = len(g2_unique_queries)
        unique_score2 = 0.0
        for query in g2_unique_queries:
            best_rank = 0
            for direction in att_diff_uniques2:
                rankings = torch.sort(torch.abs(direction.mean(0)), descending=True)[1]
                cur_rank = 1/(rankings.tolist().index(query) + 1)
                best_rank = max(cur_rank, best_rank)

            unique_score2 += best_rank

        unique_score2 = unique_score2/n2_queries if n2_queries > 0 else 0.0



        return torch.stack(overlap_direction_scores).mean().item(), unique_score1, unique_score2
                
    @torch.no_grad()
    def evaluate_entropy(self, model, generator, truncate_val=None) -> float:
        """Evaluates the model

        Returns:
            (float) accuracy (on a 0 to 1 sc ale)

        """

        bs = 128
        # Set to eval
        generator.eval()
        model.eval()

        iters = int(self.iterations/(100*bs)) + 1

        # Loop
        def cat(x1,x2): return torch.cat([x1, x2],dim=0)

        #import pdb; pdb.set_trace()
        entropy_by_direction = []
        for k in tqdm(range(self.total_directions)):
            att_diff_list = []
            for i in tqdm(range(iters)):
                z_orig = generator.sample_latent(bs)
                z_d_k = model.forward_single(z_orig, k)
                if truncate_val is not None:
                    z_orig = generator.truncate_w(z_orig, truncate_val)
                    z_d_k  = generator.truncate_w(z_d_k, truncate_val)


                #RETURNS A NORMAL IMAGE
                all_stats = self.att_classifier(generator(cat(z_orig, z_d_k)))
                orig_stats, d_stats = torch.split(all_stats, [z_orig.shape[0],z_d_k.shape[0]])
                att_diff = torch.abs(orig_stats - d_stats)
                att_diff_probs = F.softmax(att_diff, dim=1)
                att_diff_entropy = -(att_diff_probs * torch.log(att_diff_probs)).mean(1)
                att_diff_list.extend(att_diff_entropy)

            entropy_by_direction.append(torch.stack(att_diff_list).mean())

        #import pdb; pdb.set_trace()
        score = torch.stack(entropy_by_direction).mean().item() #0.03784 conv1, 0.0299 robust
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

        return torch.sigmoid(x)
        