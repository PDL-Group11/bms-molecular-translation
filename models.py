import torch
import torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from swin_transformer import SwinTransformer

from collections import OrderedDict
import argparse

def get_model(arg, pretrained=False):
    if arg.backbone == "resnet_50":
        backbone = torchvision.models.resnet50(pretrained=pretrained).features
        # TODO: Need to check the number of output channels
        backbone.out_channels = 2048
    elif arg.backbone == "mobilenet_v2":
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        backbone.out_channels = 1280
    elif arg.backbone == "swin_transformer":
        backbone = SwinTransformer()
        # TODO: Need to check the number of output channels
    
    anchor = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    if arg.model == "fast_rcnn":
        model = FasterRCNN(backbone=backbone,
                            num_classes = arg.num_classes,
                            rpn_anchor_generator = anchor,
                            box_roi_pool = roi_pooler)
    elif arg.model == "mask_rcnn":
        model = MaskRCNN(backbone=backbone,
                            num_classes = arg.num_classes,
                            rpn_anchor_generator = anchor,
                            box_roi_pool = roi_pooler)
        
        
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fast_rcnn")
    parser.add_argument("--backbone", default="mobilenet_v2")
    parser.add_argument("--num_classes", default=2)

    arg = parser.parse_args()
    model = get_model(arg)
    print(model)