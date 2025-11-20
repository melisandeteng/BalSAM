import os
from pathlib import Path
import numpy as np
import torch
from dataset import SAMTreesDataset
from geodataset.aggregator import Aggregator
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor
from shapely import Polygon
from skimage.measure import find_contours
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


def infer_test(
    sam_checkpoint="/network/scratch/XXXX-1/XXXX-2/sam_ckpt/sam_vit_h_4b8939.pth",
    root_path="/network/projects/trees-co2/dataset/",
    height_prompts_path="/network/projects/trees-co2/experiments/segmate_height_prompts/",
    fold="valid",
    output_path="/network/projects/trees-co2/sam_automatic_height_prompts_nms_test_sbl/",
):

    model_type = "vit_h"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)

    val_dataset = SAMTreesDataset(
        fold=fold, root_path=root_path, height_prompts_path=height_prompts_path
    )

    print("FILES USED", val_dataset.cocos_detected)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )
    print("Number of tiles", len(val_dataloader))

    counter = 0

    for i, elem in tqdm(enumerate(val_dataloader)):

        img, label = elem
        print("image", i)
        try:
            img = img.squeeze(0)
            predictor.set_image(np.array(img).transpose(1, 2, 0))

            points = label["prompts"].squeeze(0)
            masks, confidence, _ = predictor.predict_torch(
                point_coords=points.unsqueeze(1).cuda(),
                point_labels=torch.ones(label["prompts"].shape[1]).unsqueeze(-1).cuda(),
                multimask_output=False,
            )

            masks = masks.squeeze(1)
            polygons = [mask_to_polygon(mask.cpu()) for mask in masks]
            polygons_scores = confidence

            tiles_paths = [Path(label["image_path"][0])]

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
                scores=[polygons_scores.cpu()],
                classes=[0 for i in range(len(polygons))],
                scores_weights=None,
                score_threshold=0.5,
                nms_threshold=0.5,
                nms_algorithm="iou",
            )
        except:
            print("Problem", label["image_path"][0])
        counter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='Path to the root folder where tiles are')
    parser.add_argument('height_prompts_path', help='Path where to save the predictions')
    parser.add_argument('--fold', default="test", help="Split fold")
    parser.add_argument('--output_path', help="Path where to save predictions")
    args = parser.parse_args()

    sam_checkpoint = "/network/projects/trees-co2/sam_vit_h_4b8939.pth"

    # indicate root path to tilerized dataset
    root_path = Path(
       args.root_path
    )
    height_prompts_path = Path(
        args.height_prompts_path
    )
    fold = "test"
    
    infer_test(
        sam_checkpoint,
        root_path,
        height_prompts_path,
        args.fold,
        output_path=args.output_path
    )
