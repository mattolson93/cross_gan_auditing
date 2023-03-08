from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn

#self.att_classifier = AttClsModel("resnet18").cuda().eval()
#self.att_classifier.load_state_dict(torch.load('/usr/WS2/olson60/research/latentclr/att_classifier.pt'))
        
import torchvision.transforms as transforms
from torchvision import models


class DreModel(torch.nn.Module):
    def __init__(self, size, do_avg_pool=False, do_softplus=True):
        super(DreModel, self).__init__()

        self.model1 = create_mlp(2,size, size*2,1, batchnorm=False)
        self.model2 = create_mlp(2,size, size*2,1, batchnorm=False)
        self.size = size
        self.softplus = torch.nn.Softplus()
        self.do_pool = do_avg_pool
        self.do_softplus = do_softplus
        self.pool = nn.AvgPool2d(4) 


    @torch.cuda.amp.autocast()
    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.pool(x.reshape(-1,self.size,4,4)) if self.do_pool else x
        x = torch.flatten(x, 1)
        
        logits1 = dclamp(self.model1(x),min=-50, max=50)
        logits2 = dclamp(self.model2(x),min=-50, max=50)

        if self.do_softplus:
            logits1 = self.softplus(logits1)
            logits2 = self.softplus(logits2)

        return torch.cat([logits1,logits2],dim=1)



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

        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        self.resnet_resize = None #transforms.Resize(128)
        self.val_loss = []  # max_len == 2*k
        self.fc = torch.nn.Linear(hidden_size, 40)
        self.dropout = torch.nn.Dropout(0.2)
        self.layers = layers
        self.is_dre = is_dre
        if is_dre: exit("is_dre should never be true?")
        self.hidden_size = hidden_size

    def get_size(self): return self.hidden_size

    def preprocess_img(self, img):
        if self.is_dre: return img

        return self.do_preprocess(img)
        #typical_img = dclamp((0.5 * (img + 1)), min=0, max=1)
        #return self.resnet_resize(self.normalize(typical_img))


    def backbone_forward(self, x):
        #x = self.preprocess_img(x)
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
        #breakpoint()

        x = self.backbone.avgpool(x.reshape(-1,self.hidden_size,4,4))

        return x

    def forward(self, input, labels=None):
        #import pdb; pdb.set_trace()
        x = self.backbone_forward(input)
        #x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if self.is_dre:
            x = self.dropout(x)
            x = self.fc(x)

        return x
        
import hydra
#'/usr/WS2/olson60/research/latentclr/att_classifier.pt'    
def create_resnet(name="resnet18", layers=4, load_path=""):
    ret_transform = None
    if "voynov" in name:
        model = AttClsModel("resnet18", 5, is_pretrained=False)
        model.backbone.conv1 = nn.Conv2d( 6, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(model.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        return model.cuda(), ret_transform

    assert 0 <= layers <= 5
    if "advbn" in load_path:
        model = advbn_resnet()
        sd = torch.load(hydra.utils.to_absolute_path(load_path))['state_dict']
        for key in list(sd.keys()):
            sd[key.replace('module.', '')] = sd.pop(key)
        model.load_state_dict(sd)
        return model.cuda().eval(), ret_transform
    else:
        model = AttClsModel(name, layers)


    if load_path != "": 

        print("loading model from path: ")
        print(load_path)
        if 'att_class' in load_path and name != 'resnet50': 
            exit("att classifier needs resnet50")
        elif 'att_class' in load_path:
            ret_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        model.load_state_dict(torch.load(hydra.utils.to_absolute_path(load_path)))


    return model.cuda().eval(), ret_transform


def create_dre_model(size, do_avg_pool=False): 
    return DreModel(size, do_avg_pool).cuda()

def create_bce_model(size, do_avg_pool=False): 
    return DreModel(size, do_avg_pool, do_softplus=False).cuda()    




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



from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, momentum=0.1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.clean_bn1 = norm_layer(planes, momentum=momentum)
        self.adv_bn1 = norm_layer(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.clean_bn2 = norm_layer(planes, momentum=momentum)
        self.adv_bn2 = norm_layer(planes, momentum=momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, tag='clean'):
        identity = x

        out = self.conv1(x)
        if tag == 'clean':
            out = self.clean_bn1(out)
        else:
            out = self.adv_bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if tag == 'clean':
            out = self.clean_bn2(out)
        else:
            out = self.adv_bn2(out)

        if self.downsample is not None:
            identity = self.downsample.conv(x)
            if tag == 'clean':
                identity = self.downsample.clean_bn(identity)
            else:
                identity = self.downsample.adv_bn(identity)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, momentum=0.1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.clean_bn1 = norm_layer(width, momentum=momentum)
        self.adv_bn1 = norm_layer(width, momentum=momentum)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.clean_bn2 = norm_layer(width, momentum=momentum)
        self.adv_bn2 = norm_layer(width, momentum=momentum)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.clean_bn3 = norm_layer(planes * self.expansion, momentum=momentum)
        self.adv_bn3 = norm_layer(planes * self.expansion, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, tag='clean'):
        identity = x

        out = self.conv1(x)
        if tag == 'clean':
            out = self.clean_bn1(out)
        else:
            out = self.adv_bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if tag == 'clean':
            out = self.clean_bn2(out)
        else:
            out = self.adv_bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if tag == 'clean':
            out = self.clean_bn3(out)
        else:
            out = self.adv_bn3(out)

        if self.downsample is not None:
            identity = self.downsample.conv(x)
            if tag == 'clean':
                identity = self.downsample.clean_bn(identity)
            else:
                identity = self.downsample.adv_bn(identity)

        out += identity
        out = self.relu(out)

        return out
        

class Head(nn.Module):
    def __init__(self, layers, cut, num_classes):
        super(Head, self).__init__()
        block_num = []
        for i, layer in enumerate(layers):
            block_num.append(len(layer))
            for j, block in enumerate(layer):
                setattr(self, 'layer{}_{}'.format(str(int(cut)+i+1), str(j)), block)
        self.layer_num = len(layers)
        self.block_num = block_num
        self.cut = cut
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x, tag='clean'):
        for i in range(self.layer_num):
            for j in range(self.block_num[i]):
                m = getattr(self, 'layer{}_{}'.format(str(int(self.cut)+i+1), str(j)))
                x = m(x, tag)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class ResNet(nn.Module):
    '''
        Additional args:
            train_all (bool): Fix the first few layers during training if train_all==False. 
            cut (int): the index of layer that cut between 'feature extractor (g^{1, l})'(input -> layer{cut-1}) 
                       and 'downstream layers (g^{l+1, L})'(layer{cut} -> output)
    '''
    def __init__(self, block, layers, num_classes=1000, train_all=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, cut=2):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if num_classes == 1000:
            conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif num_classes == 10:
            conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        bn1 = norm_layer(self.inplanes)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer_list = []
        layer_list.append(self._make_layer(block, 64, layers[0]))
        layer_list.append(self._make_layer(block, 128, layers[1], stride=2,
                          dilate=replace_stride_with_dilation[0]))
        layer_list.append(self._make_layer(block, 256, layers[2], stride=2,
                          dilate=replace_stride_with_dilation[1]))
        layer_list.append(self._make_layer(block, 512, layers[3], stride=2,
                          dilate=replace_stride_with_dilation[2]))
        self.cut = cut

        # first few layers (g^{1, l}) for extracting features
        feature_x = []
        feature_x.append(conv1)
        feature_x.append(bn1)
        feature_x.append(relu)
        if num_classes == 1000:
            feature_x.append(maxpool)
        for i in range(self.cut):
            layer = layer_list[i]
            feature_x.append(nn.Sequential(*layer))
        self.feature_x = nn.Sequential(*feature_x)

        # fix the g^{1, l} when training with advbn
        if not train_all:
            for params in getattr(self, 'feature_x').parameters():
               params.requires_grad = False
            for m in self.feature_x.modules():
               if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                   m.eval()

        # downstream layers for deeper feature extraction and classification
        head = []
        if self.cut + 1 <= 4:
            for i in range(self.cut, 4):
                layer = layer_list[i]
                head.append(layer)
        self.head = Head(head, cut, num_classes)

        self.hidden_size=2048

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        #self.resnet_resize = None #transforms.Resize(128)
        #self.resize = transforms.CenterCrop(224)
        #self.resnet_resize = transforms.CenterCrop(224)

    def get_size(self): return self.hidden_size

    '''def preprocess_img(self, img):
        typical_img = dclamp((0.5 * (img + 1)), min=0, max=1)
        return self.resnet_resize(self.normalize(typical_img))'''

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', conv1x1(self.inplanes, planes * block.expansion, stride)),
                ('clean_bn', norm_layer(planes * block.expansion)),
                ('adv_bn', norm_layer(planes * block.expansion)),
            ]))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return layers

    def _forward_impl(self, x, stage='head', tag='clean'):
        x = self.feature_x(x)
        x = self.head(x, tag)
        return x

    def forward(self, x):
        return self._forward_impl(x)
        

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    return ResNet(block, layers, **kwargs)

def advbn_resnet(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, cut=1)


from torch.nn.modules.batchnorm import BatchNorm2d


def convert_ckpt(model, weight, adv_key="adv_bn"):
    """
    Args:
        model (torch.nn.Module): a model with additional adv_bn layers
        weight (state_dict): the statedict without adv_bn layers
        adv_key (str): the keyword for adv_bn layers in the model
    """
    sd = model.state_dict()
    kv = weight

    values = []
    kv_keys = []
    for k, value in kv.items():
        kv_keys.append(k)
        values.append(value)

    nkv = OrderedDict()
    index = 0
    for k in sd:
        if adv_key in k:
            # initialize the adv_bn layer using statistcs from its clean counterparts
            nkv[k] = nkv[k.replace(adv_key, adv_key.replace("adv", "clean"))]
        elif adv_key.replace("adv", "clean") in k or 'feature_x.1' in k or \
            isinstance(getattr(model, '.'.join(k.split('.')[:-1])), BatchNorm2d):  
            # in case bn stats are stored in different order (wt/bias, mean/var vs. mean/var, wt/bias) 
            if ('weight' in k and 'weight' not in kv_keys[index]) or \
                ('bias' in k and 'bias' not in kv_keys[index]):
                nkv[k] = values[index+2]
                index += 1
            elif ('mean' in k and 'mean' not in kv_keys[index]) or \
                ('var' in k and 'var' not in kv_keys[index]):
                nkv[k] = values[index-2]
                index += 1
            elif "num_batches_tracked" in k and "num_batched_tracked" not in kv_keys[index]:
                # for converting models weights from older pytorch version
                nkv[k] = torch.BoolTensor([True])
            else:
                nkv[k] = values[index]
                index += 1
        elif "num_batches_tracked" in k and "num_batched_tracked" not in kv_keys[index]:
            # for converting models weights from older pytorch version
            nkv[k] = torch.BoolTensor([True])
        else:
            nkv[k] = values[index]
            index += 1

    return nkv