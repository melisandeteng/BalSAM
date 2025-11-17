import os
from pathlib import Path

import cv2
import numpy as np
import torch
from dataset import SAMTreesDataset
from geodataset.aggregator import Aggregator
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from shapely import Polygon
from skimage.measure import find_contours
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm
import argparse


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


def get_ground_truth_viz(image, target):
    """
    image: image tensor
    target: segmentation prediction target from maskRCNN dictionary with "boxed", "masks", "label"
    """
    im = draw_segmentation_masks(
        image.cpu(), torch.Tensor(target.cpu()).type(torch.bool)
    )
    return im


def get_preds_viz(image, preds, scores, threshold=0):
    """
    image: image tensor
    preds: segmentation prediction from maskRCNN dictionary with "boxed", "masks", "label"
    threshold: threshold of score under which predictions will be ignored
    """
    indices = torch.where(scores > threshold)
    preds = preds[indices]
    im = draw_segmentation_masks(image.cpu(), preds.cpu())
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


def infer(
    sam_checkpoint="/network/projects/trees-co2/sam_vit_h_4b8939.pth",
    root_path="/network/projects/trees-co2/dataset/",
    fold="valid",
    log_comet=True,
    points_per_side=32,
    output_path="/network/projects/trees-co2/sam_automatic_preds_test_nms/",
    comet_project_name="trees",
):

    model_type = "vit_h"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam, points_per_side, box_nms_thresh=0.5, pred_iou_thresh=0.6
    )

    val_dataset = SAMTreesDataset(fold=fold, root_path=root_path)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )
    # TODO: extended summary include IoU computation taking for each ground truth the argmax over the predictions.

    for i, elem in tqdm(enumerate(val_dataloader)):

        img, label = elem

        img = img.squeeze(0)
        p = mask_generator.generate(np.transpose(img.numpy(), (1, 2, 0)))

        polygons = [mask_to_polygon(u["segmentation"]) for u in p]

        polygons_scores = [u["predicted_iou"] for u in p]
        tiles_paths = [Path(label["image_path"][0])]
        try:
            Aggregator.from_polygons(
                output_path=Path(
                    os.path.join(
                        output_path,
                        os.path.basename(label["image_path"][0]).split(".")[0]
                        + ".json",
                    )
                ),
                tiles_paths=tiles_paths,
                polygons=[polygons],
                scores=[polygons_scores],
                classes=[0 for i in range(len(polygons))],
                scores_weights=None,
                score_threshold=0.5,
                nms_threshold=0.5,
                nms_algorithm="iou",
            )
        except:
            #Just to keep track of 
            with open("nmsfailed_sbl_10.txt", "a") as f:
                f.write(f"{label['image_path'][0]}")
                f.write("\n")


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='Path to the root folder where tiles are')
    parser.add_argument('output_path', help='Path where to save the predictions')
    parser.add_argument('--points_per_side', default=10, help="number of points on automatic SAM grid")
    parser.add_argument('--fold', default="test", help="Split fold")
    parser.add_argument('--log_comet', action='store_true')
    parser.add_argument('--comet_project_name', default="sam_automatic_quebec_trees", help="comet project name")
    args = parser.parse_args()

    sam_checkpoint = "/network/projects/trees-co2/sam_vit_h_4b8939.pth"
    root_path = Path(args.root_path)
    infer(
        sam_checkpoint,
        root_path,
        args.fold,
        args.log_comet,
        args.points_per_side,
        args.output_path,
        args.comet_project_name,
    )
    print("Done")
