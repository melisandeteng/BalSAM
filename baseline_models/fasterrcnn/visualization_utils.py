import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def create_colormap(num_colors):
    colors = plt.cm.hsv(np.linspace(0, 1, num_colors))
    colors = colors[:, :3]
    return colors


def get_ground_truth_viz(image, target, categories, cmap):
    """
    image: image tensor
    target: segmentation prediction target from maskRCNN dictionary with "boxed", "masks", "label"

    typically:
    CATEGORIES = utils.get_categories(dataset_name)
    CMAP = create_colormap(len(CATEGORIES))

    """
    im = draw_bounding_boxes(
        image.cpu(),
        torch.Tensor(target["boxes"].cpu()),
        [categories[int(i) - 1]["name"] for i in target["labels"].cpu()],
        colors=[
            tuple([int(j * 255) for j in cmap[i - 1]]) for i in target["labels"].cpu()
        ],
        width=2,
        font_size=12,
    )
    im = draw_segmentation_masks(
        im.cpu(),
        torch.Tensor(target["masks"].cpu()).type(torch.bool),
        0.5,
        colors=[
            tuple([int(j * 255) for j in cmap[i - 1]]) for i in target["labels"].cpu()
        ],
    )
    return im


def get_preds_viz(image, preds, threshold, categories, cmap):
    """
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
            categories[int(i) - 1]["name"] + " : " + str(scores[j].item())
            for j, i in enumerate(labels.cpu())
        ],
        colors=[tuple([int(j * 255) for j in cmap[i - 1]]) for i in labels.cpu()],
        width=2,
        font_size=12,
    )

    if "masks" in preds.keys():
        im = draw_segmentation_masks(
            im.cpu(),
            ((preds["masks"]).cpu() > 0.5).type(torch.bool),
            0.3,
            colors=[tuple([int(j * 255) for j in cmap[i - 1]]) for i in labels.cpu()],
        )
    return im
