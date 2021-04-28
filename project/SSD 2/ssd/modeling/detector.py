from torch import nn

from ssd.modeling.backbone.resnet18 import Model
from ssd.modeling.backbone.resnext50 import ResNeXt50
from ssd.modeling.backbone.resnet18_800x450 import ResNet18_800x450
from ssd.modeling.backbone.resnet18_600x600 import ResNet18_600x600
from ssd.modeling.backbone.resnet50_600x600 import ResNet50_600x600
from ssd.modeling.backbone.resnext50_600x600 import ResNeXt50_600x600
from ssd.modeling.backbone.resnext50_800x450 import ResNeXt50_800x450

from ssd.modeling.box_head.box_head import SSDBoxHead
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd import torch_utils
import importlib

import torchvision.models as models


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = SSDBoxHead(cfg)
        print(
            "Detector initialized. Total Number of params: ",
            f"{torch_utils.format_params(self)}")
        print(
            f"Backbone number of parameters: {torch_utils.format_params(self.backbone)}")
        print(
            f"SSD Head number of parameters: {torch_utils.format_params(self.box_head)}")

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME

    
    if backbone_name == "resNet":
        model = Model(cfg)
        return model
    if backbone_name == "resNet18":
        model = models.resnet18(cfg)
        return model
    if backbone_name == "resnet50_600x600":
	    model = ResNet50_600x600(cfg)
	    return model
    if backbone_name == "resneXt50_600x600":
	    model = ResNeXt50_600x600(cfg)
	    return model
    if backbone_name == "resnet18_600x600":
        model = ResNet18_600x600(cfg)
        return model
    if backbone_name == "resnet18_800x450":
        model = ResNet18_800x450(cfg)
        return model
    if backbone_name == "resnext50":
        model = ResNeXt50(cfg)
        return model
    if backbone_name == "resnext50_800x450":
        model = ResNeXt50_800x450(cfg)
        return model
    
