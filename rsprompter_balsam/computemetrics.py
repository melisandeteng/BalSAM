import pickle
import base64
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import numpy as np
import pandas as pd
import argparse
import os

def convert_preds_pickle_to_coco(file, 
                                save_path= "predictions_coco_format.json"):
    """
    Convert predictions pickle file into json in COCO format
    """
    with open(file, "rb") as f:
        data = pickle.load(f)
    
    coco_predictions = []

    for pred in data:  # Adjust based on your pickle structure
        for i in range(pred['pred_instances']["labels"].shape[0]):  # Replace 'objects' with the correct key
            mask = pred['pred_instances']['masks'][i]
            mask["counts"] = mask["counts"].decode("utf-8")
            coco_predictions.append({
                "image_id": pred['img_id'],  # Image ID
                "category_id": (pred['pred_instances']['labels'][i] + 1).item(),  # Category ID
                "segmentation": mask,  # Segmentation in RLE or polygon
                "bbox": pred['pred_instances']['bboxes'][i].tolist(),  # Bounding box [x, y, width, height]
                "score": pred['pred_instances']['scores'][i].item(),  # Confidence score
            })

    with open(save_path, "w") as f:
        json.dump(coco_predictions, f)


    
def main(preds_json= "predictions_coco_format.json",  annots = "/DATA_ROOT/quebec_trees_tiles_fullresolution/merged_annots_test_new.json" , save_path_csv = "predictions.csv"):
    """
    compute mAP and mIoU single class and multiclass mAP. 
    """
    print(f"evaluation of {preds_json}")
    
    with open(preds_json, "r") as f:
        coco_preds = json.load(f)
        
    img_id_to_preds = {}
    for i, pred in enumerate(coco_preds):
        img_id = pred["image_id"]
        if img_id not in img_id_to_preds:
            img_id_to_preds[img_id] = [i]
        else:
            img_id_to_preds[img_id] += [i]
    # Load COCO ground truth
    coco_gt = COCO(annots)
    
    iou_list = []
    metric = MeanAveragePrecision(iou_type= "segm", class_metrics=True)
    metric_single = MeanAveragePrecision(iou_type="segm", class_metrics = True)
    
    preds_classes = []
    tar_classes = []
    for im_id in coco_gt.getImgIds():
        targets = []
        targets_single = []
        ann_ids = coco_gt.getAnnIds(imgIds=im_id)
        anns = coco_gt.loadAnns(ann_ids)

        # Extract boxes and labels
        boxes = []
        labels = []
        masks = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])
            if 'counts' in ann["segmentation"] and isinstance(ann["segmentation"]['counts'], str):
                counts = base64.b64decode(ann["segmentation"]['counts'])
                ann["segmentation"]['counts'] = counts

            masks.append(maskUtils.decode(ann["segmentation"]))
        labels= np.array(labels)
        masks = np.array(masks)
        targets.append({"boxes": boxes, "labels": torch.Tensor(labels).type(torch.int),"masks": torch.Tensor(masks).type(torch.uint8)})
        targets_single.append({"boxes": boxes, "labels": torch.full_like(torch.Tensor(labels), 0).type(torch.int),"masks": torch.Tensor(masks).type(torch.uint8)})
        
        dict_pred = {"boxes": [], "scores": [], "labels": [], "masks":[]}
        dict_single = {"boxes": [], "scores": [], "labels": [], "masks":[]}
        if im_id in img_id_to_preds:
            for pred_idx in img_id_to_preds[im_id]:

                pred = coco_preds[pred_idx]
                x, y, w, h = pred['bbox']
                dict_pred["boxes"].append([x, y, x + w, y + h])
                dict_pred["scores"].append(pred["score"])
                dict_pred["labels"].append(pred["category_id"])

                dict_pred["masks"].append(maskUtils.decode(pred["segmentation"]))
            dict_pred["labels"] = torch.Tensor(dict_pred["labels"]).type(torch.int)
            dict_pred["masks"] = torch.Tensor(np.array(dict_pred["masks"])).type(torch.uint8)
            dict_pred["scores"] = torch.Tensor(dict_pred["scores"])
            
            dict_single["masks"]= dict_pred["masks"] 
            dict_single["scores"]= dict_pred["scores"] 
            dict_single["boxes"]= dict_pred["boxes"] 
            dict_single["labels"]= torch.full_like( dict_pred["labels"] , 0).type(torch.int)
        else:
            #empty predictions for this image
            print(f"no prediction for {im_id}")
            dict_pred = {"boxes": [], "scores": torch.Tensor([]), "labels": torch.Tensor([]), "masks":torch.Tensor([])}
            dict_single = {"boxes": [], "scores": torch.Tensor([]), "labels": torch.Tensor([]), "masks":torch.Tensor([])}
        
        metric.update([dict_pred], targets)
        metric_single.update([dict_single], targets_single)
        
        preds = [dict_pred]
        iou_per_instance = []
        for k in range(len(targets)):
            for idxt, t in enumerate(targets[k]["masks"]):
                pred_class = -1
                max_iou = 0
                tar_class = targets[k]["labels"][idxt]
    
                for idx, p in enumerate(preds[k]["masks"]):
                    intersection = (t & p).sum().float()
                    union = (t | p).sum().float()
                    iou = float(intersection / union) if union > 0 else 0.0
                    if iou > max_iou:
                        max_iou=iou
                        pred_class = preds[k]["labels"][idx]
                iou_per_instance.append(max_iou) 
                tar_classes.append(tar_class.item())
                if pred_class == -1:
                    preds_classes.append(pred_class)
                else:   
                    preds_classes.append(pred_class.item())
                    
        iou_list.append(np.mean(iou_per_instance))
            
    a = metric.compute()
    print(a)
    b = metric_single.compute()
    
    print(b)    
    print("Iou", np.mean(iou_list))
    df = pd.DataFrame(
    {'preds': preds_classes,
     'targets': tar_classes
    })
    print("saving df")
    df.to_csv(save_path_csv)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('picklefile', help='predictions pickle file from infer')
    parser.add_argument('--save-file', help='json file')
    
    parser.add_argument('--annots', help='annotations json file')
    parser.add_argument(
        '--csv_path',
        type=str,
        help='path to csv to save preds and targets classes to compute confusion matrices')
    args = parser.parse_args()
    return args

if __name__=="__main__":

    args = parse_args()
    
    file = args.picklefile
    save_path = args.save_file
    
    annots = args.annots
    save_path_csv = os.path.dirname(str(args.save_file))
    convert_preds_pickle_to_coco(file, save_path)
    main(save_path, annots, save_path_csv + "predictions_targets.csv")
