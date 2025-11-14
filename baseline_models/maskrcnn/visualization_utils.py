import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

CATEGORIES = [
    {"id": 1, "name": "piba", "supercategory": "pinus"},
    {"id": 2, "name": "pima", "supercategory": "picea"},
    {"id": 3, "name": "pist", "supercategory": "pinus"},
    {"id": 4, "name": "pigl", "supercategory": "picea"},
    {"id": 5, "name": "thoc", "supercategory": "thuja"},
    {"id": 6, "name": "ulam", "supercategory": "ulmaceae"},
    {
        "id": 7,
        "name": "other",
        "other_names": ["qeru", "bepa", "potr", "lala", "pire"],
        "supercategory": "",
    },
    {"id": 8, "name": "beal", "supercategory": "betula"},
    {"id": 9, "name": "acsa", "supercategory": "acer"},
]


def create_colormap(num_colors):
    colors = plt.cm.hsv(np.linspace(0, 1, num_colors))
    colors = colors[:, :3]
    return colors


CMAP = create_colormap(len(CATEGORIES))


def get_ground_truth_viz(image, target):
    """
    Produces a visualization of the rgb image with ground truth labels.
    image: image tensor
    target: segmentation prediction target from maskRCNN dictionary with "boxed", "masks", "label"
    """
    im = draw_bounding_boxes(
        image.cpu(),
        torch.Tensor(target["boxes"].cpu()),
        [CATEGORIES[int(i) - 1]["name"] for i in target["labels"].cpu()],
        colors=[
            tuple([int(j * 255) for j in CMAP[i - 1]]) for i in target["labels"].cpu()
        ],
        width=2,
        font_size=12,
    )
    im = draw_segmentation_masks(
        im.cpu(),
        torch.Tensor(target["masks"].cpu()).type(torch.bool),
        0.5,
        colors=[
            tuple([int(j * 255) for j in CMAP[i - 1]]) for i in target["labels"].cpu()
        ],
    )
    return im


def get_preds_viz(image, preds, threshold=0):
    """
    Produces a visualization of the rgb image with predicted labels.
    image: image tensor
    preds: segmentation prediction from maskRCNN dictionary with "boxed", "masks", "label"
    threshold: threshold of score under which predictions will be ignored
    """
    indices = torch.where(preds["scores"] > threshold)
    scores = preds["scores"][indices].detach()
    labels = preds["labels"][indices]
    im = draw_bounding_boxes(
        image.cpu(),
        torch.Tensor(preds["boxes"][indices].cpu()),
        [
            CATEGORIES[int(i) - 1]["name"] + " : " + str(scores[j].item())
            for j, i in enumerate(labels.cpu())
        ],
        colors=[tuple([int(j * 255) for j in CMAP[i - 1]]) for i in labels.cpu()],
        width=2,
        font_size=12,
    )
    im = draw_segmentation_masks(
        im.cpu(),
        torch.Tensor((preds["masks"].squeeze(1)[indices].cpu() > 0.5)).type(torch.bool),
        0.3,
        colors=[tuple([int(j * 255) for j in CMAP[i - 1]]) for i in labels.cpu()],
    )
    return im
