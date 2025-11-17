import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
import comet_ml
import numpy as np
import omegaconf
import torch
import torchvision
import torchvision.transforms.functional as F
from model import get_model_detection, get_sam
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from train_fasterrcnn import collate_fn_detection as collate_fn_detection_base
from train_fasterrcnn import get_dataloader as get_dataloader_base
from train_fasterrcnn_dsm import \
    collate_fn_detection as collate_fn_detection_dsm
from train_fasterrcnn_dsm import get_dataloader as get_dataloader_dsm
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
    # config.save_ckpt_path = config.log_dir + f"_{config.seed}"
    # config.load_ckpt_path=os.path.join(config.save_ckpt_path, "best_model.pt")
    # print(omegaconf.OmegaConf.to_yaml(config))
    return config


def infer_and_save(config, dsm=False):
    """
    inference over the test set, save predictions and compute Mean Average Precision and mIoU
    """
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
    model = get_model_detection(config.num_classes, config.model_version).to(device)
    sam_model, sam_processor = get_sam(config.sam_version)
    sam_model = sam_model.to(device)

    print("resuming from " + config.load_ckpt_path)
    checkpoint = torch.load(config.load_ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406, 0], image_std=[0.229, 0.224, 0.225, 1])
    # model.transform = grcnn
    epoch_start = checkpoint["epoch"]
    print(f"Resuming {checkpoint['epoch']}")

    if dsm:
        test_dataloader = get_dataloader_dsm(
            config, config.json_test, "test", collate_fn_detection_dsm
        )
    else:
        test_dataloader = get_dataloader_base(config, "test", collate_fn_detection_base)

    # test_dataloader = get_dataloader_base(config, "test", collate_fn_detection_base)

    preds_save_path = os.path.join(Path(config.load_ckpt_path).parent, "preds_test/")
    print("Saving predictions to ", preds_save_path)
    os.makedirs(preds_save_path, exist_ok=True)

    # save mAP and IoU
    map_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    map_metric_single = MeanAveragePrecision(iou_type="segm", class_metrics=False)
    iou_list = []
    with torch.no_grad():

        # Validation steps
        model.eval()

        print(f"Using threshold score {config.threshold_score}")

        for i, (im, tar) in tqdm(enumerate(test_dataloader)):

            im = im.to(device)
            out = model(im)
            preds = []
            preds_single = []
            for j, outputs in enumerate(out):
                # Threshold on NMS
                indices = torchvision.ops.nms(
                    outputs["boxes"], outputs["scores"], config.nms_score
                )
                # Threshold on probabilities
                indices = torch.where(
                    outputs["scores"][indices] > config.threshold_score
                )
                # Select best predictions
                labels = outputs["labels"][indices]
                outputs["labels"] = labels
                boxes = outputs["boxes"][indices]
                outputs["boxes"] = boxes
                scores = outputs["scores"][indices]
                outputs["scores"] = scores
                if len(indices[0]) == 0:
                    # No prediction to keep = no SAM
                    outputs["masks"] = torch.empty((0, 1024, 1024))
                    outputs["masks_scores"] = deepcopy(outputs["scores"])
                else:
                    try:
                        im_size = list(im[j].shape[1:])
                        # height, width = im_size
                        # input_sam_boxes = [[b[0] / width, b[1] / height, b[2] / width, b[3] / height]
                        #                    for b in boxes.tolist()]
                        input_sam_boxes = [boxe.tolist() for boxe in boxes]
                        if config.with_dsm:
                            input_im = im[j][:3, ...]
                        else:
                            input_im = im[j]
                        sam_inputs = sam_processor(
                            input_im,
                            input_boxes=[input_sam_boxes],
                            return_tensors="pt",
                            do_rescale=False,
                        ).to(device)
                        sam_outputs = sam_model(**sam_inputs, multimask_output=False)
                        masks_scores = sam_outputs["iou_scores"].squeeze(0, 2)
                        masks = sam_outputs["pred_masks"]
                        masks = sam_processor.image_processor.post_process_masks(
                            sam_outputs.pred_masks.cpu(),
                            sam_inputs["original_sizes"].cpu(),
                            sam_inputs["reshaped_input_sizes"].cpu(),
                        )
                        masks = [
                            mask.squeeze((0, 2)).type(torch.bool) for mask in masks
                        ]
                        outputs["masks"] = torch.stack(masks).squeeze(0, 2)
                        outputs["masks_scores"] = masks_scores
                    except IndexError:
                        pass
                # Compute the overall score as the average of det + seg scores
                # We must have the same number of scores!
                assert len(outputs["scores"]) == len(
                    outputs["masks_scores"]
                ), "Must have the same number of boxes and masks scores"
                # Avoid mixing classif and seg scores ???

                # if len(outputs["masks_scores"]) != 0:
                #    outputs["scores"] = (outputs["scores"] + outputs["masks_scores"]) / 2.

                # else:
                # No prediction to keep = no SAM = no score
                #    outputs["scores"] = deepcopy(outputs["boxes_scores"])

                im_id = tar[j]["image_id"]
                name = os.path.basename(im_id).strip(".tif")
                dict_save = {k: outputs[k].cpu() for k in list(outputs.keys())}

                # save predictions
                # with open(os.path.join(preds_save_path, f"{name}.pkl"), "wb") as f:
                #    pickle.dump(dict_save, f)

                # dict_save["masks"] = (dict_save["masks"].squeeze(1) >0.5).type(torch.uint8)
                dict_save["masks"] = outputs["masks"]
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
                for t in tar[k]["masks"]:
                    max_iou = 0
                    for p in preds[k]["masks"]:
                        intersection = (t & p).sum().float()
                        union = (t | p).sum().float()
                        iou = float(intersection / union) if union > 0 else 0.0
                        max_iou = max(max_iou, iou)
                    iou_per_instance.append(max_iou)

            iou_list.append(np.mean(iou_per_instance))

    # print(map_metric.compute())
    print("Saved predictions")
    map_ = map_metric.compute()
    map_single = map_metric_single.compute()
    return (preds_save_path, map_, map_single, np.mean(iou_list))


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

    preds_save_path, map_, map_single, iou = infer_and_save(config, config.with_dsm)

    print("evaluate model ")
    print(map_, map_single, iou)


if __name__ == "__main__":
    # python inference.py config.yaml seed=0
    main()
