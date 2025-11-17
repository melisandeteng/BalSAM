"""
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from transformers import SamModel, SamProcessor


def get_model_detection(num_classes, model_version="resnet50"):

    if model_version == "resnet50":
        # load an instance segmentation model pre-trained on COCO
        # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model = resnet_fpn_backbone(
            backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT
        )
        model = FasterRCNN(model, num_classes)
    elif model_version == "resnet50scratch":
        # load an instance segmentation model pre-trained on COCO
        # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        model = resnet_fpn_backbone(backbone_name="resnet50", weights=None)
        model = FasterRCNN(model, num_classes)
    elif model_version == "resnet50dsm":
        # load an instance segmentation model pre-trained on COCO
        # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model = resnet_fpn_backbone(
            backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT
        )
        model = FasterRCNN(model, num_classes)
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406, 0],
            image_std=[0.229, 0.224, 0.225, 1],
        )
        model.transform = grcnn
        old_weights = model.backbone.body.conv1.weight.data[:, :3, :, :].clone()
        model.backbone.body.conv1 = nn.Conv2d(
            4,
            model.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.backbone.body.conv1.weight.data[:, :3, :, :] = old_weights
    else:
        raise ValueError("model name for Faster R-CNN is not handled.")

    return model


def get_sam(sam_version):
    model = SamModel.from_pretrained(sam_version).to("cuda")
    processor = SamProcessor.from_pretrained(sam_version)
    return model, processor
