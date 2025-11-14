import os
import pickle
import random
import sys
from copy import deepcopy
from pathlib import Path

import comet_ml
import numpy as np
import omegaconf
import tifffile
import torch
import torchvision.transforms.functional as F
from model import get_model_instance_segmentation
from model_sam import get_segment_anything
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from train_maskrcnn import \
    collate_fn_segmentation as collate_fn_segmentation_base
from train_maskrcnn import get_dataloader as get_dataloader_base
from train_maskrcnn_dsm import \
    collate_fn_segmentation as collate_fn_segmentation_dsm
from train_maskrcnn_dsm import get_dataloader as get_dataloader_dsm
from utils import get_categories
from visualization_utils import (create_colormap, get_ground_truth_viz,
                                 get_preds_viz)


def get_config(config_path="config.yaml", cli_config=None, scaler=None):
    """
    Parses config.yaml and changes checkpoint to load to best model.
    """
    # train on the GPU or on the CPU, if a GPU is not available

    config = omegaconf.OmegaConf.load(config_path)
    if cli_config is not None:
        config = omegaconf.OmegaConf.merge(config, cli_config)
    config.save_ckpt_path = config.save_ckpt_path + f"_{config.seed}"
    # config.load_ckpt_path=os.path.join(config.save_ckpt_path, "best_model.pt")
    print(omegaconf.OmegaConf.to_yaml(config))
    return config


def infer_and_save(config, dsm=False):

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    # get the model using our helper function
    model = get_model_instance_segmentation(
        config.num_classes, config.model_version
    ).to(device)
    seg_anything_model = get_segment_anything()

    print("resuming from " + config.load_ckpt_path)
    checkpoint = torch.load(config.load_ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch_start = checkpoint["epoch"]
    print(f"Resuming {epoch_start}")

    if dsm:
        test_dataloader = get_dataloader_dsm(
            config, config.json_test, "test", collate_fn_segmentation_dsm
        )
    else:
        test_dataloader = get_dataloader_base(
            config, "test", collate_fn_segmentation_base
        )

    preds_save_path = os.path.join(
        Path(config.load_ckpt_path).parent, "preds_test_meli_sam_boxesboxes_thresh/"
    )
    print("Saving predictions to ", preds_save_path)
    os.makedirs(preds_save_path, exist_ok=True)
    map_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    map_metric_single = MeanAveragePrecision(iou_type="segm", class_metrics=False)
    iou_list = []

    with torch.no_grad():

        # Validation steps
        model.eval()
        for i, (im, tar) in enumerate(test_dataloader):
            im2 = im.to(device)
            out = model(im2)
            preds = []
            preds_single = []

            outputs_new = {"masks": [], "sam_scores": []}
            for j, outputs in enumerate(out):
                # Threshold on NMS
                # indices = torchvision.ops.nms(outputs["boxes"], outputs["scores"], 0.5)
                # Threshold on probabilities
                indices = torch.where(outputs["scores"] > config.threshold_score)
                # Select best predictions
                # Update best labels
                labels = outputs["labels"][indices]
                outputs["labels"] = labels
                # Update best boxes
                boxes = outputs["boxes"][indices]
                outputs["boxes"] = boxes
                # Update best masks
                masks = outputs["masks"][indices]
                outputs["masks"] = masks
                # Update best scores
                scores = outputs["scores"][indices]
                outputs["scores"] = scores

                if len(indices[0]) == 0:
                    # No prediction to keep = no SAM
                    outputs["sam_scores"] = deepcopy(outputs["scores"])
                else:
                    # try:
                    im_size = list(im[j].shape[1:])

                    height, width = im_size
                    img = tifffile.imread(tar[j]["image_id"])[:, :, :3]
                    # input_im = np.transpose(input_im, (1,2,0))
                    seg_anything_model.set_image(img)

                    for k in range(len(outputs["labels"])):

                        input_sam_box = boxes[k].tolist()
                        seg_any_boxes = torch.Tensor(input_sam_box).unsqueeze(0)
                        input_sam_masks = (masks[k] > 0.5).type(torch.uint8)
                        input_sam_masks = input_sam_masks[0] * 255
                        seg_any_masks = (
                            input_sam_masks[None, :, :].type(torch.float).cuda()
                        )
                        seg_any_masks = seg_any_masks.resize_((1, 256, 256))

                        seg_anything_outputs = seg_anything_model.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=seg_any_boxes.to("cuda"),
                            mask_input=seg_any_masks.type(torch.float).cuda(),
                            multimask_output=False,
                        )
                        seg_any_masks = seg_anything_outputs[0].squeeze(1)
                        seg_any_scores = seg_anything_outputs[1].squeeze(1)
                        outputs_new["masks"].append(seg_any_masks)
                        outputs_new["sam_scores"].append(seg_any_scores[0])

                    outputs["masks"] = torch.stack(outputs_new["masks"])
                    outputs["sam_scores"] = torch.Tensor(
                        outputs_new["sam_scores"]
                    ).cuda()
                # except:
                #    print("no pred")
                #    pass

                # assert len(outputs["scores"]) == len(outputs["sam_scores"]), "Must have the same number of boxes and masks scores"
                # Avoid mixing classif and seg scores??
                if len(outputs["sam_scores"]) != 0:
                    outputs["scores"] = (
                        outputs["scores"] + outputs["sam_scores"]
                    ) / 2.0
                else:
                    # No prediction to keep = no SAM = no score
                    outputs["scores"] = deepcopy(outputs["scores"])

                im_id = tar[j]["image_id"]
                name = os.path.basename(im_id).strip(".tif")
                dict_save = {k: outputs[k].detach().cpu() for k in list(outputs.keys())}

                if config.comet:  # don't log all the images, it's too much

                    fig1 = get_ground_truth_viz(
                        torch.Tensor(im[j] * 255).type(torch.uint8),
                        tar[j],
                        categories,
                        cmap,
                    )
                    fig2 = get_preds_viz(
                        torch.Tensor(im[j] * 255).type(torch.uint8),
                        dict_save,
                        threshold=config.threshold_score,
                        categories=categories,
                        cmap=cmap,
                    )
                    fig = torch.cat((fig1, fig2), -1)
                    image_log = F.to_pil_image(fig)
                    experiment.log_image(image_log, name)

                # save predictions
                with open(os.path.join(preds_save_path, f"{name}.pkl"), "wb") as f:
                    pickle.dump(dict_save, f)

                # dict_save["masks"] = (dict_save["masks"].squeeze(1) >0.5).type(torch.uint8)
                dict_save["masks"] = outputs["masks"].squeeze(1)
                preds.append(dict_save)

                dict_save2 = deepcopy(dict_save)
                dict_save2["labels"] = torch.zeros_like(dict_save2["labels"])
                preds_single.append(dict_save2)

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
                for t in tar[k]["masks"]:
                    max_iou = 0
                    t = t.cuda()

                    for p in preds[k]["masks"]:
                        p = p.cuda()
                        intersection = (t & p).sum().float()
                        union = (t | p).sum().float()
                        iou = float(intersection / union) if union > 0 else 0.0
                        max_iou = max(max_iou, iou)
                    iou_per_instance.append(max_iou)

            iou_list.append(np.mean(iou_per_instance))

    map_ = map_metric.compute()
    map_single = map_metric_single.compute()
    iou = np.mean(iou_list)

    return (map_, map_single, iou)


def main():
    config_file = sys.argv[1]
    cli_config = omegaconf.OmegaConf.from_dotlist(sys.argv[2:])
    config = get_config(config_file, cli_config)

    config.nms_score = 1
    config.with_dsm = False
    config.batch_size = 1
    map_, map_single, iou = infer_and_save(config, dsm=False)
    print(map_, map_single, iou)


if __name__ == "__main__":
    main()
