import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

def resnet18_builder(pretrained=False,**kwargs):
    if pretrained==False:
        model=resnet18()
    else:
        model=resnet18(ResNet18_Weights.IMAGENET1K_V1)
    return model 

def resnet50_builder(pretrained=False,**kwargs):
    if pretrained==False:
        model=resnet50()
    else:
        model=resnet50(ResNet50_Weights.IMAGENET1K_V2)
    return model 


def build_resnet(model_name='resnet18',pretrained=True):
    if model_name=='resnet18':
        model=resnet18_builder(pretrained)
        feat_dim=512
    if model_name=='resnet50':
        model=resnet50_builder(pretrained)
        feat_dim=2048
    backbone=nn.Sequential(*list(model.children())[:-2])
    return backbone,feat_dim







