import json
import os
from pathlib import Path
import argparse
import comet_ml
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from dataset import SAMTreesDataset
from geodataset.utils import \
    coco_rle_segmentation_to_mask as rle_segmentation_to_mask
from shapely import Polygon
from skimage.measure import find_contours
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_segmentation_masks


def mask_to_polygon(mask: np.ndarray, simplify_tolerance: float = 1.0) -> Polygon:
    """
    Converts a 1HW mask to a simplified shapely Polygon by finding the contours of the mask
    and simplifying it.

    Parameters
    ----------
    mask: np.ndarray
        The mask to convert, in 1HW format.
    simplify_tolerance: float
        The tolerance for simplifying the polygon. Higher values result in more simplified shapes.

    Returns
    -------
    Polygon
     A simplified shapely Polygon object representing the outer boundary of the mask.
    """
    # Ensure mask is 2D
    if mask.ndim != 2:
        raise ValueError("Mask must be in HW format (2D array).")

    # Pad the mask to avoid boundary issues
    padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)

    # Find contours on the mask, assuming mask is binary
    contours = find_contours(padded_mask, 0.5)

    if len(contours) == 0:
        # returning empty, dummy polygon at 0,0
        return Polygon([(0, 0), (0, 0), (0, 0)])

    # Take the longest contour as the main outline of the object
    longest_contour = max(contours, key=len)

    # Convert contour coordinates from (row, column) to (x, y)
    # and revert the padding added to the mask
    longest_contour_adjusted_xy = [(y - 1, x - 1) for x, y in longest_contour]

    # Convert contour to Polygon
    polygon = Polygon(longest_contour_adjusted_xy)

    # Simplify the polygon
    simplified_polygon = polygon.simplify(
        tolerance=simplify_tolerance, preserve_topology=True
    )

    return simplified_polygon



def get_viz(image, target):
    """
    image: image tensor
    target: segmentation prediction target from maskRCNN dictionary with "boxed", "masks", "label"
    """
    im = draw_segmentation_masks(
        image.cpu(), torch.Tensor(target.cpu()).type(torch.bool)
    )
    return im


def load_image(image_path: str) -> np.ndarray:
    """
    Loads the image.

    Args:
        image_path (str): The path to the image.

    Returns:
        image (numpy.ndarray): The loaded image.
    """
    # loading the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def compute_metrics(
    root_path="/network/projects/trees-co2/dataset/",
    height_prompts_path="/network/projects/trees-co2/experiments/segmate_height_prompts/",
    fold="valid",
    output_path="/network/projects/trees-co2/sam_automatic_height_prompts_nms_test/",
    log_comet=False,
    comet_project_name="trees"
):
    """
    compute metrics from predictions obtained with with infer_new_nms script for the SAM+DSM prompts model
    """

    if log_comet:
        if os.environ.get("COMET_API_KEY"):
            experiment = comet_ml.Experiment(
                api_key=os.environ.get("COMET_API_KEY"), project_name=comet_project_name
            )

        else:
            print("no COMET API Key found..continuing without logging..")
            # return

    val_dataset = SAMTreesDataset(
        fold=fold, root_path=root_path, height_prompts_path=height_prompts_path
    )

    print("FILES USED", val_dataset.cocos_detected)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )
    print("Number of tiles", len(val_dataloader))
    map_metric_single_class = MeanAveragePrecision(
        iou_type="segm", class_metrics=True, extended_summary=True
    )
    # TODO: extended summary include IoU computation taking for each ground truth the argmax over the predictions.

    for i, elem in enumerate(val_dataloader):

        img, label = elem
        img = img.squeeze(0)

        tile_name = os.path.basename(label["image_path"][0]).split(".")[0]
        annots = Path(os.path.join(output_path, tile_name + ".json"))
        if os.path.exists(annots):
            with open(annots, "rb") as f:
                ann = json.load(f)

            # import pdb; pdb.set_trace()
            masks = torch.Tensor(
                np.array(
                    [
                        rle_segmentation_to_mask(u["segmentation"])
                        for u in ann["annotations"]
                    ]
                )
            )
            masks = masks.type(torch.uint8)

            scores = torch.Tensor(
                [u["other_attributes"]["score"] for u in ann["annotations"]]
            )
            gt_masks = label["masks"].squeeze(0)

            map_metric_single_class.update(
                [
                    {
                        "labels": torch.zeros(masks.shape[0], dtype=torch.uint8),
                        "masks": masks,
                        "scores": scores,
                    }
                ],
                [
                    {
                        "labels": torch.zeros(gt_masks.shape[0], dtype=torch.uint8),
                        "masks": gt_masks,
                    }
                ],
            )

            fig1 = get_viz(img, gt_masks)  # (img.squeeze(0), gt_masks)
            fig2 = get_viz(img, masks)  # (img.squeeze(0), masks)
            fig = torch.cat((fig1, fig2), -1)
            image_log = F.to_pil_image(fig)
            experiment.log_image(image_log, tile_name)

    map_single_score = map_metric_single_class.compute()
    print("mAP single class test", map_single_score["map"])

    ious = map_single_score["ious"]

    # ious: torchmetrics mAP extended summary ious which is a dictionary containing the IoU values for every image/class combination e.g. ious[(0,0)] would contain the IoU for image 0 and class 0. Each value is a tensor with shape (n,m) where n is the number of detections and m is the number of ground truth boxes for that image/class combination."""

    total_iou = 0
    num_instances = 0
    for elem in range(len(val_dataloader)):
        if (elem, 0) in ious:
            iou = ious[(elem, 0)]

            values, _ = iou.max(axis=0)
            try:
                num_instances += len(values)

                total_iou += values.sum()
            except:
                print(values)

                num_instances += 1
                total_iou += values.sum()
        else:
            print("error")
            print(elem)

    print("mIoU val single class trees", total_iou / num_instances)

    experiment.log_metric("mAP single class test", map_single_score["map"])
    experiment.log_metric("mIoU val single class trees", total_iou / num_instances)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='Path to the root folder where tiles are')
    parser.add_argument('height_prompts_path', help='Path where to save the predictions')
    parser.add_argument('--fold', default="test", help="Split fold")
    parser.add_argument('--output_path', help="Path where to save predictions")
    parser.add_argument('--log_comet', action='store_true')
    parser.add_argument('--comet_project_name', default="sam_automatic_quebec_trees", help="comet project name")
    args = parser.parse_args()

    sam_checkpoint = "/network/projects/trees-co2/sam_vit_h_4b8939.pth"
    root_path = Path(
       args.root_path
    ) 
    height_prompts_path = Path(
        args.height_prompts_path
    ) 
    fold = "test"
    compute_metrics(
        root_path,
        height_prompts_path,
        args.fold,
        output_path=args.output_path,
        log_comet=args.log_comet,
        comet_project_name=args.comet_project_name
        
    )
