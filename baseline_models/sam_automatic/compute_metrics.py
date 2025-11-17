import json
import os
from pathlib import Path
import comet_ml
import numpy as np
import torch
import torchvision.transforms.functional as F
from dataset import SAMTreesDataset
from geodataset.utils import \
    coco_rle_segmentation_to_mask as rle_segmentation_to_mask
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm
import argparse

def get_viz(image, target):
    """
    image: image tensor
    target: segmentation prediction target from maskRCNN dictionary with "boxed", "masks", "label"
    """
    im = draw_segmentation_masks(
        image.cpu(), torch.Tensor(target.cpu()).type(torch.bool)
    )
    return im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='Path to the root folder where tiles are')
    parser.add_argument('preds_path', help='Path where to save the predictions')
    parser.add_argument('--points_per_side', default=10, help="number of points on automatic SAM grid")
    parser.add_argument('--fold', default="test", help="Split fold")
    parser.add_argument('--log_comet', action='store_true')
    parser.add_argument('--comet_project_name', default="sam_automatic_quebec_trees", help="comet project name")
    args = parser.parse_args()

    fold = "test"
    root_path = args.root_path
    pred_path = args.preds_path
    val_dataset = SAMTreesDataset(fold=args.fold, root_path=root_path)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    if args.log_comet:
        if os.environ.get("COMET_API_KEY"):
            experiment = comet_ml.Experiment(
                api_key=os.environ.get("COMET_API_KEY"), project_name=args.comet_project_name
            )

        else:
            print("no COMET API Key found..continuing without logging..")

    map_metric_single_class = MeanAveragePrecision(
        iou_type="segm", class_metrics=True, extended_summary=True
    )

    

    for i, elem in tqdm(enumerate(val_dataloader)):

        img, labels = elem

        im_path = labels["image_path"][0]
        tile_name = os.path.basename(im_path).split(".")[0]

        annots = os.path.join(pred_path, tile_name + ".json")
        if os.path.exists(annots):
            with open(annots, "rb") as f:
                ann = json.load(f)

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
                np.array([u["other_attributes"]["score"] for u in ann["annotations"]])
            )
            gt_masks = labels["masks"].squeeze(0)

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
            fig1 = get_viz(img.squeeze(0), gt_masks)
            fig2 = get_viz(img.squeeze(0), masks)
            fig = torch.cat((fig1, fig2), -1)
            image_log = F.to_pil_image(fig)
            experiment.log_image(image_log, tile_name)

    # map_metric_single_class.compute()
    map_single_score = map_metric_single_class.compute()
    print(map_single_score["map"])
    print(map_single_score.keys())
    ious = map_single_score["ious"]
    # ious: torchmetrics mAP extended summary ious which is a dictionary containing the IoU values for every image/class combination e.g. ious[(0,0)] would contain the IoU for image 0 and class 0. Each value is a tensor with shape (n,m) where n is the number of detections and m is the number of ground truth boxes for that image/class combination."""
    print("mAP single class test", map_single_score["map"])

    total_iou = 0
    num_instances = 0
    for i, elem in enumerate(range(len(val_dataloader))):

        iou = ious[(elem, 0)]

        values, _ = iou.max(axis=0)
        try:
            num_instances += len(values)
        except:
            num_instances += 1
        total_iou += values.sum()

    print("mIoU val single class trees", total_iou / num_instances)
