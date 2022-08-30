import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        
        # our_backbone
        resnet = resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        # self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        
        # self.f = nn.Sequential(*self.f)
        # self.f = resnet50()
        # projection head
        self.g = nn.Sequential(nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True))
        self.r = nn.Sequential(nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), nn.Linear(256, 4, bias=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        feature = torch.flatten(x, start_dim=1)
        feature_1, feature_2 = torch.split(feature,256,1)
        out = self.g(feature_1)
        rotate = self.r(feature_2)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), rotate
