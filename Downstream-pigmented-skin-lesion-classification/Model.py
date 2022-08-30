from torchvision import models
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import torch


# 如果feature_extract = False，则对模型进行微调，并更新所有模型参数
# 如果feature_extract = True，则仅更新最后一层参数，其余参数保持不变. called freezing
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        # pretrained：直接加载pre-train模型中预先训练好的参数
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)  # 修改最后一层分类器个数

    elif model_name == "resnet50":
        """resnet
        """
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # 修改最后一层分类器个数

    elif model_name == "resnet18":
        """resnet
        """
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # 修改最后一层分类器个数

    else:
        print("Invalid model name, exiting...")
        exit()
    return model