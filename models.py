import torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from swin_transformer import SwinTransformer

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
                            aspect_ratio=((0.5, 0.1, 2.0),))

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