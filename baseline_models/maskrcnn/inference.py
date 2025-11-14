import json
import os
import pickle
import random
import sys
from copy import deepcopy

import comet_ml
import numpy as np
import omegaconf
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as F
from model import get_model_instance_segmentation
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from train_maskrcnn import \
    collate_fn_segmentation as collate_fn_segmentation_base
from train_maskrcnn import get_dataloader as get_dataloader_base
from train_maskrcnn_dsm import \
    collate_fn_segmentation as collate_fn_segmentation_dsm
from train_maskrcnn_dsm import get_dataloader as get_dataloader_dsm
from train_maskrcnn_dsm_bci import get_dataloader as get_dataloader_dsm_bci
from train_maskrcnn_encode_dsm import \
    get_dataloader as get_dataloader_encode_dsm
from utils import get_categories
from visualization_utils import (create_colormap, get_ground_truth_viz,
                                 get_preds_viz)


def get_config(config_path="config.yaml", cli_config=None, scaler=None):
    """Parses config.yaml and changes checkpoint to load to best model."""
    # train on the GPU or on the CPU, if a GPU is not available

    config = omegaconf.OmegaConf.load(config_path)
    if cli_config is not None:
        config = omegaconf.OmegaConf.merge(config, cli_config)

    config.save_ckpt_path = config.save_ckpt_path + f"_{config.seed}"
    if config.load_ckpt_path == "" or config.load_ckpt_path is None:
        config.load_ckpt_path = os.path.join(config.save_ckpt_path, "best_model.pt")

    print(omegaconf.OmegaConf.to_yaml(config))
    return config


def infer_and_save(config, dsm=False, save_csv_preds_tar_classes=False):
    """Inference over the test set, save predictions and compute Mean Average
    Precision and mIoU."""

    if config.comet:
        if os.environ.get("COMET_API_KEY"):

            experiment = comet_ml.Experiment(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name=config.comet_project,
            )
            if config.comet_exp_name:
                experiment.set_name(config.comet_exp_name)
            if config.comet_tags:
                experiment.add_tags(
                    omegaconf.OmegaConf.to_object(config.comet_tags)
                    + [f"seed_{config.seed}"]
                )
        categories = get_categories(config.dataset)
        cmap = create_colormap(len(categories))

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    # get the model using our helper function
    model = get_model_instance_segmentation(config.num_classes, config.model_version)
    # move model to the right device
    model.to(device)

    print("resuming from " + config.load_ckpt_path)

    checkpoint = torch.load(config.load_ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if config.model_version == "resnet50dsmgrads":
        grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406, 0, 0],
            image_std=[0.229, 0.224, 0.225, 1, 1],
        )
        model.transform = grcnn

    epoch_start = checkpoint["epoch"]
    print(f"Resuming {epoch_start}")

    if dsm:
        if config.dataset == "bci":
            test_dataloader = get_dataloader_dsm_bci(
                config, "test", collate_fn_segmentation_base
            )
        if config.model_version == "resnet50encodedsm":
            test_dataloader = get_dataloader_encode_dsm(
                config, config.json_test, "test", collate_fn_segmentation_dsm
            )
        else:
            test_dataloader = get_dataloader_dsm(
                config, config.json_test, "test", collate_fn_segmentation_dsm
            )
    else:
        test_dataloader = get_dataloader_base(
            config, "test", collate_fn_segmentation_base
        )

    preds_save_path = os.path.join(config.save_ckpt_path, "preds_test_0/")
    print("Saving predictions to ", preds_save_path)
    os.makedirs(preds_save_path, exist_ok=True)

    # save MaP and IoU
    map_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    map_metric_single = MeanAveragePrecision(iou_type="segm", class_metrics=False)
    iou_list = []
    with torch.no_grad():

        # Validation steps
        model.eval()

        print(f"Using threshold score {config.threshold_score}")

        preds_classes = []
        tar_classes = []

        for i, (im, tar) in enumerate(test_dataloader):

            im = im.to(device)
            out = model(im)
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in out]

            preds = []
            preds_single = []
            for j, dict_pred in enumerate(outputs):
                im_id = tar[j]["image_id"]
                name = os.path.basename(im_id).strip(".tif")
                mask = dict_pred["scores"] > config.threshold_score
                dict_save = {
                    k: dict_pred[k][mask].cpu() for k in list(dict_pred.keys())
                }

                # save predictions

                with open(os.path.join(preds_save_path, f"{name}.pkl"), "wb") as f:
                    pickle.dump(dict_save, f)

                dict_save["masks"] = (dict_save["masks"].squeeze(1) > 0.5).type(
                    torch.uint8
                )
                preds.append(dict_save)

                dict_save2 = deepcopy(dict_save)
                dict_save2["labels"] = torch.zeros_like(dict_save2["labels"])
                preds_single.append(dict_save2)

                if config.comet:  # don't log all the images, it's too much

                    fig1 = get_ground_truth_viz(
                        torch.Tensor(im[j][:3, :, :] * 255).type(torch.uint8),
                        tar[j],
                        categories,
                        cmap,
                    )
                    fig2 = get_preds_viz(
                        torch.Tensor(im[j][:3, :, :] * 255).type(torch.uint8),
                        dict_save,
                        threshold=config.threshold_score,
                        categories=categories,
                        cmap=cmap,
                    )
                    fig = torch.cat((fig1, fig2), -1)
                    image_log = F.to_pil_image(fig)
                    experiment.log_image(image_log, name)

            for elem in tar:
                elem["boxes"] = torch.Tensor(elem["boxes"])
                elem.pop("image_id")

            map_metric.update(preds, tar)

            tar2 = deepcopy(tar)
            for elem in tar2:
                elem["labels"] = torch.zeros_like(elem["labels"])

            map_metric_single.update(preds_single, tar2)

            iou_per_instance = []

            for k in range(len(tar)):
                for idxt, t in enumerate(tar[k]["masks"]):
                    pred_class = -1
                    max_iou = 0
                    tar_class = tar[k]["labels"][idxt]
                    for idx, p in enumerate(preds[k]["masks"]):
                        intersection = (t & p).sum().float()
                        union = (t | p).sum().float()
                        iou = float(intersection / union) if union > 0 else 0.0
                        max_iou = max(max_iou, iou)
                        pred_class = preds[k]["labels"][idx]

                    iou_per_instance.append(max_iou)
                    if save_csv_preds_tar_classes:
                        tar_classes.append(tar_class.item())
                        preds_classes.append(pred_class.item())

            iou_list.append(np.mean(iou_per_instance))

    print(map_metric.compute())
    print("Saved predictions", preds_save_path)
    map_ = map_metric.compute()
    map_single = map_metric_single.compute()

    # if you want to save the predicted classes and target classes to compute confusion matrices
    if save_csv_preds_tar_classes:
        df = pd.DataFrame({"preds": preds_classes, "targets": tar_classes})
        df.to_csv(os.path.join(preds_save_path, "preds_tars_classes.csv"))

    return (preds_save_path, map_, map_single, np.mean(iou_list))


def get_metrics_from_saved_preds(config, preds_save_path, dsm=False):

    if dsm:
        test_dataloader = get_dataloader_dsm(
            config, config.json_test, "test", collate_fn_segmentation_dsm
        )
    else:
        print("get dataloader")
        test_dataloader = get_dataloader_base(
            config, "test", collate_fn_segmentation_base
        )

    map_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    iou_list = []

    for i, (im, tar) in enumerate(test_dataloader):
        preds = []
        for j in range(len(tar)):
            im_id = tar[j]["image_id"]
            if dsm:
                name = test_dataloader.dataset.data["images"][im_id]["file_name"]
            else:

                name = os.path.basename(im_id).strip(".tif")

            with open(os.path.join(preds_save_path, name + ".pkl"), "rb") as f:
                pred = pickle.load(f)
            if i == 1 or i == 2:
                print(os.path.join(preds_save_path, name + ".pkl"))
            pred["masks"] = (pred["masks"].squeeze(1) > 0.5).type(torch.uint8)

            preds.append(pred)

            # del(pred)
        for elem in tar:
            elem["boxes"] = torch.Tensor(elem["boxes"])
            elem.pop("image_id")

        map_metric.update(preds, tar)
        print(map_metric.compute()["map"])

        iou_per_instance = []
        for k in range(len(tar)):
            for t in tar[k]["masks"]:
                max_iou = 0
                for p in preds[k]["masks"]:
                    intersection = (t & p).sum().float()
                    union = (t | p).sum().float()
                    iou = float(intersection / union) if union > 0 else 0.0
                    max_iou = max(max_iou, iou)
                iou_per_instance.append(max_iou)
        iou_list.append(np.mean(iou_per_instance))

    map_ = map_metric.compute()
    iou = np.mean(iou_list)

    return (map_, iou)


def get_stats_per_class(
    annotations_file="/network/projects/trees-co2/final_tiles/merged_annots_test_new.json",
):
    with open(annotations_file, "rb") as f:
        annots = json.load(f)
    count_per_species = {}
    for annot in annots["annotations"]:
        if annot["category_id"] in count_per_species.keys():
            count_per_species[annot["category_id"]] += 1
        else:
            count_per_species[annot["category_id"]] = 1
    return count_per_species


def main():
    config_file = sys.argv[1]

    cli_config = omegaconf.OmegaConf.from_dotlist(sys.argv[2:])
    config = get_config(config_file, cli_config)
    preds_save_path = os.path.join(config.save_ckpt_path, "preds_test_meli/")
    print(f"saved preds to {preds_save_path}")

    preds_save_path, map_, map_single, iou = infer_and_save(config, dsm=True)

    print(map_, map_single, iou)


if __name__ == "__main__":

    # python test_save_preds.py config.yaml seed=0

    main()
