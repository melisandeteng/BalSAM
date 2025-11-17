import io
import math
import os
import random
import sys
from pathlib import Path
import albumentations as a
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torchvision
import torchvision.transforms.functional as F
import wandb
from dataset import TreesDSMDataset
from engine import evaluate
from model import get_model_detection, get_sam
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from utils import get_categories
from visualization_utils import (create_colormap, get_ground_truth_viz,
                                 get_preds_viz)


def show_matched_masks(
    image: np.ndarray,
    masks: np.ndarray,
    additional_masks=None,
    size: int = None,
    save_path=None,
) -> None:
    """
    Shows the masks.

    Args:
        image (numpy.ndarray): The image to be shown.
        masks (numpy.ndarray): The mask overlayed image to be shown.
        additional_masks (list(numpy.ndarray)): Additional masks to be shown, usually for comparison.
        size (int): The size of the plot.
    """

    # Define the number of plots and their arrangement

    im = Image.fromarray(image)

    rows = masks.shape[0]
    if additional_masks is not None:
        cols = 3
    else:
        cols = 2
    # Create subplots
    fig, axes = plt.subplots(rows, cols)  # Adjust figsize as needed
    if rows == 1:
        axes[0].imshow(im)
        mask_set = masks[0, :, :]
        axes[1].imshow(im)  # Overlay image 1
        axes[1].imshow(mask_set, cmap="viridis", alpha=0.4)

        if additional_masks is not None:
            gt_set = additional_masks[0, :, :]
            ax = axes[2]
            ax.imshow(im)  # Overlay image 1
            ax.imshow(gt_set, cmap="viridis", alpha=0.4)
    else:
        axes[0, 0].imshow(im)

        # Plot each subplot
        try:
            for i in range(rows):

                mask_set = masks[i, :, :]

                ax = axes[i, 1]
                ax.imshow(im)  # Overlay image 1
                ax.imshow(mask_set, cmap="viridis", alpha=0.4)

                if additional_masks is not None:
                    gt_set = additional_masks[i, :, :]
                    ax = axes[i, 2]
                    ax.imshow(im)  # Overlay image 1
                    ax.imshow(gt_set, cmap="viridis", alpha=0.4)
            for ax in axes.flat:
                ax.axis("off")
        except:
            print("number rows", rows)
            print("masks_shape", masks.shape)
            print("gt_shape", additional_masks.shape)
            return 0
    # Adjust layout

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_pil = Image.open(buf)
    image_np = np.array(image_pil)
    plt.close(fig)
    return image_np


def collate_fn_detection(batch):
    return (
        torch.stack([elem[0] for elem in batch]),
        tuple([elem[1] for elem in batch]),
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config, json_file, fold, collate_fn=collate_fn_detection):
    if fold == "train":
        transform = a.Compose(
            [a.HorizontalFlip(p=0.5), a.VerticalFlip(p=0.5)],
            bbox_params=a.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )
    else:
        transform = None

    dataset = TreesDSMDataset(
        Path(json_file), dsm_normalization=config.dsm_normalization, transform=transform
    )
    # Define percentage of dataset to use
    if fold in ("train", "val", "test"):
        num_samples = int(len(dataset) * config.percentage)

        # Randomly select a subset
        dataset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples])
    print("dataset_length", len(dataset))
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=fold == "train",
        num_workers=0,
        collate_fn=collate_fn_detection,
    )
    return data_loader


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def get_iou_score(map_score, img_to_label_dict):
    """ious: torchmetrics mAP extended summary ious which is a dictionary containing
    the IoU values for every image/class combination e.g. ious[(0,0)] would contain the
    IoU for image 0 and class 0. Each value is a tensor with shape (n,m) where n is the
    number of detections and m is the number of ground truth boxes for that image/class
    combination."""
    ious = map_score["ious"]
    total_iou = 0
    num_instances = 0
    for elem in img_to_label_dict:
        lab = img_to_label_dict[elem]
        for l in np.unique(lab):
            iou = ious[(elem, l)]
            if iou == []:
                # no instance was predicted for the class, IoU is 0 for every gt instance of that class
                num_instances += lab.count(l)
            else:
                values, _ = iou.max(axis=0)
                try:
                    num_instances += len(values)
                    total_iou += values.sum()
                except:
                    num_instances += 1  # there is just one instance
                    total_iou += values.item()
    iou_score = total_iou / num_instances
    return iou_score


def train(config_path="config.yaml", cli_config=None, scaler=None):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = omegaconf.OmegaConf.load(config_path)
    if cli_config is not None:
        config = omegaconf.OmegaConf.merge(config, cli_config)

    if config.wandb:
        wandb_id = wandb.util.generate_id()
        wandb.init(
            dir=config.log_dir,
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=dict(config),
            id=wandb_id,
            resume="allow",
        )
        id_expe = wandb.run.name
    else:
        id_expe = str(np.random.randint(10000)).zfill(5)
    config.log_dir = config.log_dir + f"{id_expe}"

    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)
    elif config.load_ckpt_path != "":
        # load a specific checkpoint using its name
        config.load_ckpt_path = os.path.join(config.log_dir, config.load_ckpt_path)
    else:
        # there was at least a checkpoint saved, load the best
        if len(os.listdir(config.log_dir)) != 0:
            config.load_ckpt_path = os.path.join(config.log_dir, "best_model.pt")

    print(omegaconf.OmegaConf.to_yaml(config))
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    categories = get_categories(config.dataset)
    cmap = create_colormap(len(categories))

    train_dataloader = get_dataloader(
        config, config.json_train, "train", collate_fn_detection
    )
    val_dataloader = get_dataloader(
        config, config.json_val, "val", collate_fn_detection
    )
    test_dataloader = get_dataloader(
        config, config.json_test, "test", collate_fn_detection
    )

    # get the model using our helper function
    print("Model version", config.model_version)
    model = get_model_detection(config.num_classes, config.model_version).to(device)
    sam_model, sam_processor = get_sam(config.sam_version)
    sam_model = sam_model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable parameters", sum(p.numel() for p in params))

    optimizer = torch.optim.Adam(
        params, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.exponentialLR_gamma
    )

    epoch_start = 0
    if config.load_ckpt_path != "" and config.load_ckpt_path is not None:
        print("resuming from " + config.load_ckpt_path)
        checkpoint = torch.load(config.load_ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        #loss = checkpoint["loss"]
        print(f"Starting at epoch {checkpoint['epoch']}")

    print("train_loader_length", len(train_dataloader))
    print("val_loader_length", len(val_dataloader))
    print("test_loader_length", len(test_dataloader))

    val_score = 0
    for epoch in range(epoch_start, config.num_epochs):

        # train for one epoch, printing every 10 iterations
        running_training_loss = 0
        model.train(True)

        for images, targets in tqdm(train_dataloader):
            images = images.to(device)  # list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if lr_scheduler is not None and (epoch + 1) % config.scheduler_step == 0:
                lr_scheduler.step()

            running_training_loss += loss_value
        epoch_training_loss = running_training_loss / len(train_dataloader)
        print(f"epoch training loss {epoch}", epoch_training_loss)

        if epoch % config.save_every_epochs == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_training_loss,
                },
                os.path.join(config.log_dir, f"fasterrnn_epoch_{epoch}.pt"),
            )

        with torch.no_grad():
            print(f"Validation loop epoch {epoch}")
            # Validation steps
            map_metric_det = MeanAveragePrecision(
                iou_type="bbox", class_metrics=True, extended_summary=True
            )
            map_metric_seg = MeanAveragePrecision(
                iou_type="segm", class_metrics=True, extended_summary=True
            )
            map_metric_single_class_det = MeanAveragePrecision(
                iou_type="bbox", class_metrics=True
            )
            map_metric_single_class_seg = MeanAveragePrecision(
                iou_type="segm", class_metrics=True
            )
            # TODO: extended summary include IoU computation taking for each ground truth the argmax over the predictions.

            model.eval()

            img_to_label_dict = {}
            counter = 0

            rand_batch = np.random.randint(
                len(val_dataloader)
            )  # For visualisation purposes
            for i, (im, tar) in enumerate(val_dataloader):

                im = im.to(device)
                out = model(im)
                targets = [
                    {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in tar
                ]
                for l, outputs in enumerate(out):
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
                    boxes = outputs["boxes"][indices]

                    u = targets[l].copy()
                    u["labels"] = torch.zeros(u["labels"].shape, dtype=torch.int)

                    # Detection metrics
                    map_metric_det.update(
                        [
                            {
                                "labels": labels,
                                "boxes": boxes,
                                "scores": outputs["scores"][indices],
                            }
                        ],
                        [targets[l]],
                    )
                    map_metric_single_class_det.update(
                        [
                            {
                                "labels": torch.zeros(labels.shape, dtype=torch.int),
                                "boxes": boxes,
                                "scores": outputs["scores"][indices],
                            }
                        ],
                        [u],
                    )
                    # keep a mapping of img index and labels
                    img_to_label_dict[counter] = list(
                        targets[l]["labels"].cpu().numpy()
                    )

                    if (epoch + 1) % config.sam_loop_evey_epochs == 0:
                        # Instance segmentation with SAM
                        try:
                            sam_boxes = [boxe.tolist() for boxe in boxes]

                            # We aexclude the DSM to feed the image to SAM
                            sam_inputs = sam_processor(
                                im[l][:3, ...],
                                input_boxes=[sam_boxes],
                                return_tensors="pt",
                                do_rescale=False,
                            ).to(device)
                            with torch.no_grad():
                                sam_outputs = sam_model(
                                    **sam_inputs, multimask_output=False
                                )

                            masks_scores = sam_outputs["iou_scores"].squeeze(0, 2)
                            masks = sam_outputs["pred_masks"]
                            masks = sam_processor.image_processor.post_process_masks(
                                sam_outputs.pred_masks.cpu(),
                                sam_inputs["original_sizes"].cpu(),
                                sam_inputs["reshaped_input_sizes"].cpu(),
                            )
                            masks = [
                                (mask.squeeze((0, 2)) > 0).type(torch.bool)
                                for mask in masks
                            ]
                            outputs["masks"] = masks
                            map_metric_seg.update(
                                [
                                    {
                                        "labels": labels,
                                        "masks": torch.stack(masks).squeeze(0, 2),
                                        "scores": masks_scores,
                                    }
                                ],
                                [targets[l]],
                            )
                            map_metric_single_class_seg.update(
                                [
                                    {
                                        "labels": torch.zeros(
                                            labels.shape, dtype=torch.int
                                        ),
                                        "masks": torch.stack(masks).squeeze(0, 2),
                                        "scores": masks_scores,
                                    }
                                ],
                                [u],
                            )
                        except IndexError:
                            pass
                    counter += 1

                    if (epoch + 1) % config.log_images_every_epochs == 0:
                        if (
                            config.wandb and i == rand_batch
                        ):  # don't log all the images, it's too much
                            for j in range(len(targets)):
                                # Exclude the DSM channel
                                fig1 = get_ground_truth_viz(
                                    torch.Tensor(im[j][:3, ...] * 255).type(
                                        torch.uint8
                                    ),
                                    targets[j],
                                    categories=categories,
                                    cmap=cmap,
                                )
                                fig2 = get_preds_viz(
                                    torch.Tensor(im[j][:3, ...] * 255).type(
                                        torch.uint8
                                    ),
                                    out[j],
                                    threshold=config.threshold_score,
                                    categories=categories,
                                    cmap=cmap,
                                )
                                fig = torch.cat((fig1, fig2), -1)
                                image_log = F.to_pil_image(fig)
                                log_dict = {"Epoch": epoch}
                                wandb_img = wandb.Image(image_log)
                                log_dict["GT - Pred"] = wandb_img
                                wandb.log(log_dict, step=epoch)

            map_score_det = map_metric_det.compute()
            map_single_score_det = map_metric_single_class_det.compute()
            if (epoch + 1) % config.sam_loop_evey_epochs == 0:
                map_score_seg = map_metric_seg.compute()
                map_single_score_seg = map_metric_single_class_seg.compute()

            if "ious" in map_score_det:
                iou_score_det = get_iou_score(map_metric_det, img_to_label_dict)
                if config.wandb:
                    log_dict = {"Epoch": epoch}
                    log_dict["Detection mIoU val single class trees"] = iou_score_det
                    wandb.log(log_dict, step=epoch)
                else:
                    print(f"Detection mIoU val single class trees: {iou_score_det}")

            if (
                epoch + 1
            ) % config.sam_loop_evey_epochs == 0 and "ious" in map_score_seg:
                iou_score_seg = get_iou_score(map_metric_seg, img_to_label_dict)
                if config.wandb:
                    log_dict = {"Epoch": epoch}
                    log_dict["Segmentation mIoU val single class trees"] = iou_score_seg
                    wandb.log(log_dict, step=epoch)
                else:
                    print(f"Segmentation mIoU val single class trees: {iou_score_seg}")

        if map_score_det["map"].item() > val_score:
            val_score = map_score_det["map"].item()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_training_loss,
                },
                os.path.join(config.log_dir, "best_model.pt"),
            )

        print("Detection mAP val:", map_score_det["map"].item())
        if (epoch + 1) % config.sam_loop_evey_epochs == 0:
            print("Segmentation mAP val:", map_score_seg["map"].item())

        if config.wandb:
            log_dict = {"Epoch": epoch}
            log_dict = {"lr": lr_scheduler.get_last_lr()[0]}
            log_dict["train_loss"] = epoch_training_loss
            log_dict["Detection mAP val"] = map_score_det["map"].item()
            log_dict["Detection mAP val (single class)"] = map_single_score_det[
                "map"
            ].item()
            if (epoch + 1) % config.sam_loop_evey_epochs == 0:
                log_dict["Segmentation mAP val"] = map_score_seg["map"].item()
                log_dict["Segmentation mAP val (single class)"] = map_single_score_seg[
                    "map"
                ].item()
            wandb.log(log_dict)
        del map_metric_det
        del map_metric_single_class_det
        if (epoch + 1) % config.sam_loop_evey_epochs == 0:
            del map_metric_seg
            del map_metric_single_class_seg
        del img_to_label_dict

    print("evaluate model val")
    evaluate(model, val_dataloader, "cuda")

    print("evaluate model test")
    evaluate(model, test_dataloader, "cuda")


def main():
    config_file = sys.argv[1]
    print(config_file)
    cli_config = omegaconf.OmegaConf.from_dotlist(sys.argv[2:])
    train(config_file, cli_config)


if __name__ == "__main__":
    main()
