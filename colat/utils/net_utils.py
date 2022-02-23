from collections import OrderedDict

import torch

#self.att_classifier = AttClsModel("resnet18").cuda().eval()
#self.att_classifier.load_state_dict(torch.load('/usr/WS2/olson60/research/latentclr/att_classifier.pt'))
        
import torchvision.transforms as transforms
from torchvision import models

class AttClsModel(torch.nn.Module):
    def __init__(self, model_type, layers, is_dre=False, is_pretrained=True):
        super(AttClsModel, self).__init__()
        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=is_pretrained)
            hidden_size = 2048
        elif model_type == 'resnet34':
            self.backbone = models.resnet34(pretrained=is_pretrained)
            hidden_size = 512
        elif model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=is_pretrained)
            hidden_size = 512
        else:
            raise NotImplementedError
        #self.lambdas = torch.ones((40,), device=device)
        #import pdb; pdb.set_trace()
        #sd = torch.load("/usr/WS2/olson60/research/latentclr/colat/utils/checkpoint_0100.pth.tar")['state_dict']
        #self.load_state_dict(sd, strict=False)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize(128)
        self.val_loss = []  # max_len == 2*k
        self.fc = torch.nn.Linear(hidden_size, 40)
        self.dropout = torch.nn.Dropout(0.2)
        self.layers = layers
        self.is_dre = is_dre

    def preprocess_img(self, img):
        if self.is_dre: return img
        typical_img = (img).clamp(0, 1)
        return self.resize(self.normalize(typical_img))


    def backbone_forward(self, x):
        x = self.preprocess_img(x)
        #import pdb; pdb.set_trace()
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

        x = self.backbone.avgpool(x.reshape(-1,512,4,4))

        return x

    def forward(self, input, labels=None):
        x = self.backbone_forward(input)
        #import pdb; pdb.set_trace()
        #x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if self.is_dre:
            x = self.dropout(x)
            x = self.fc(x)

        return x
        
import hydra
#'/usr/WS2/olson60/research/latentclr/att_classifier.pt'    
def create_resnet(name="resnet18", layers=4, load_path=None):
    assert 0 <= layers <= 5
    model = AttClsModel(name, layers).cuda().eval()
    if load_path is not None: 
        #import pdb; pdb.set_trace()
        print("loading model from path: ")
        print(load_path)
        model.load_state_dict(torch.load(hydra.utils.to_absolute_path(load_path)), strict=False)
    return model

def create_dre_model(name="resnet18", layers=4, out_preds=1):
    assert 0 <= layers <= 5

    model = AttClsModel(name, layers=5, is_dre=True, is_pretrained=False).train()

    #model.preprocess_img = torch.nn.Identity() 
    model.backbone.conv1 = torch.nn.Identity() 
    model.backbone.bn1 = torch.nn.Identity() 
    model.backbone.relu = torch.nn.Identity() 
    model.backbone.maxpool = torch.nn.Identity() 

    #

    i = 1
    if layers >= i: model.backbone.layer1 = torch.nn.Identity() 
    i+=1
    if layers >= i: model.backbone.layer2 = torch.nn.Identity() 
    i+=1
    if layers >= i: model.backbone.layer3 = torch.nn.Identity() 
    i+=1
    if layers >= i: model.backbone.layer4 = torch.nn.Identity() 
    
    #import pdb; pdb.set_trace()
    #model.fc = torch.nn.Linear(model.fc.weight.shape[1], 1)
    size = model.fc.weight.shape[1]
    model.fc = create_mlp(2,size, size*2,out_preds, batchnorm=False)

    return model.cuda()
        




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
        if batchnorm:
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
