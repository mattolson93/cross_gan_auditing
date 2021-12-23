from collections import OrderedDict

import torch

#self.att_classifier = AttClsModel("resnet18").cuda().eval()
#self.att_classifier.load_state_dict(torch.load('/usr/WS2/olson60/research/latentclr/att_classifier.pt'))
        

from torchvision import models

class AttClsModel(torch.nn.Module):
    def __init__(self, model_type, layers):
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
        self.fc = torch.nn.Linear(hidden_size, 40)
        self.dropout = torch.nn.Dropout(0.5)
        self.layers = layers

    def backbone_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        i = 0
        if self.layers == i: return x
        i+=1

        x = self.backbone.layer1(x)
        if self.layers == i: return x
        i+=1

        x = self.backbone.layer2(x)
        if self.layers == i: return x
        i+=1
        x = self.backbone.layer3(x)
        if self.layers == i: return x
        i+=1
        x = self.backbone.layer4(x)
        if self.layers == i: return x
        i+=1

        x = self.backbone.avgpool(x)

        return x

    def forward(self, input, labels=None):
        x = self.backbone_forward(input)
        #import pdb; pdb.set_trace()
        #x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.dropout(x)
        #x = self.fc(x)

        return x
        
#'/usr/WS2/olson60/research/latentclr/att_classifier.pt'    
def create_resnet(name="resnet18", layers=4, load_path=None):
    assert 0 <= layers <= 5
    model = AttClsModel(name, layers).cuda().eval()
    if load_path is not None: model.load_state_dict(torch.load(load_path))
    return model



def create_mlp(
    depth: int,
    in_features: int,
    middle_features: int,
    out_features: int,
    bias: bool = True,
    batchnorm: bool = True,
    final_norm: bool = False,
):
    # initial dense layer
    layers = []
    layers.append(
        (
            "linear_1",
            torch.nn.Linear(
                in_features, out_features if depth == 1 else middle_features
            ),
        )
    )

    #  iteratively construct batchnorm + relu + dense
    for i in range(depth - 1):
        layers.append(
            (f"batchnorm_{i+1}", torch.nn.BatchNorm1d(num_features=middle_features))
        )
        layers.append((f"relu_{i+1}", torch.nn.ReLU()))
        layers.append(
            (
                f"linear_{i+2}",
                torch.nn.Linear(
                    middle_features,
                    out_features if i == depth - 2 else middle_features,
                    False if i == depth - 2 else bias,
                ),
            )
        )

    if final_norm:
        layers.append(
            (f"batchnorm_{depth}", torch.nn.BatchNorm1d(num_features=out_features))
        )

    # return network
    return torch.nn.Sequential(OrderedDict(layers))
