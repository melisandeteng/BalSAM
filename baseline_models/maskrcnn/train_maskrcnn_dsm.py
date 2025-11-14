import io
import math
import os
import random
import sys
from pathlib import Path

import albumentations as a
import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torchvision
import torchvision.transforms.functional as F
from dataset import TreesDSMDataset
from engine import evaluate
from model import get_model_instance_segmentation
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


def collate_fn_segmentation(batch):
    return (
        torch.stack([elem[0] for elem in batch]),
        tuple([elem[1] for elem in batch]),
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config, json_file, fold, collate_fn=collate_fn_segmentation):
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
    if fold == "train":
        num_samples = int(len(dataset) * config.percentage)

        # Randomly select a subset
        dataset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples])
    print("dataset_length", len(dataset))
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=fold == "train",
        num_workers=0,
        collate_fn=collate_fn_segmentation,
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


class MaskRCNNTrainer:
    def __init__(self, model, comet_project=None):
        """
        save_path_embeddings (str): path to pkl file where to save embeddings
        """
        super().__init__()
        self.model = model

        if comet_project:
            self.comet = True
            if os.environ.get("COMET_API_KEY"):
                self.experiment = comet_ml.Experiment(
                    api_key=os.environ.get("COMET_API_KEY"), project_name=comet_project
                )

            else:
                print("no COMET API Key found..continuing without logging..")
                return
        else:
            self.comet = False


def train(config_path="config.yaml", cli_config=None, scaler=None):

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = omegaconf.OmegaConf.load(config_path)
    if cli_config is not None:
        config = omegaconf.OmegaConf.merge(config, cli_config)

    config.save_ckpt_path = config.save_ckpt_path + f"_{config.seed}"
    if not os.path.isdir(config.save_ckpt_path):
        os.makedirs(config.save_ckpt_path)
    elif config.load_ckpt_path != "" and config.load_ckpt_path is not None:
        # load a specific checkpoint using its name
        config.load_ckpt_path = os.path.join(
            config.save_ckpt_path, config.load_ckpt_path
        )
    else:
        # there was at least a checkpoint saved, load the best
        if len(os.listdir(config.save_ckpt_path)):
            config.load_ckpt_path = os.path.join(config.save_ckpt_path, "best_model.pt")
    print(omegaconf.OmegaConf.to_yaml(config))
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

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
    # torch.use_deterministic_algorithms(True)

    train_dataloader = get_dataloader(
        config, config.json_train, "train", collate_fn_segmentation
    )
    val_dataloader = get_dataloader(
        config, config.json_val, "val", collate_fn_segmentation
    )
    test_dataloader = get_dataloader(
        config, config.json_test, "test", collate_fn_segmentation
    )

    categories = get_categories(config.dataset)
    cmap = create_colormap(len(categories))
    if config.weights_loss is not None:
        weights_loss = torch.Tensor(config.weights_loss)
    else:
        weights_loss = config.weights_loss
    # get the model using our helper function
    model = get_model_instance_segmentation(
        config.num_classes, config.model_version, weights_loss, config.hierarchical_loss
    )
    # move model to the right device
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable parameters", sum(p.numel() for p in params))
    optimizer = torch.optim.SGD(
        params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )

    epoch_start = 0
    if config.load_ckpt_path != "" and config.load_ckpt_path is not None:
        print("resuming from " + config.load_ckpt_path)

        checkpoint = torch.load(config.load_ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        print(f"Starting at epoch {checkpoint['epoch']}")

    print("train_loader_length", len(train_dataloader))
    print("val_loader_length", len(val_dataloader))
    print("test_loader_length", len(test_dataloader))

    val_score = 0

    # step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9, last_epoch=-1)

    for epoch in range(epoch_start, config.num_epochs):

        # train for one epoch, printing every 10 iterations
        running_training_loss = 0
        model.train(True)

        lr_scheduler = None

        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(train_dataloader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        for images, targets in tqdm(train_dataloader):
            images = images.to(device)  # list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            with torch.amp.autocast("cuda", enabled=scaler is not None):

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)
            # loss_value = sum(loss for loss in loss_dict_reduced.values()).item()
            loss_value = losses.detach().item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                losses.backward()
                optimizer.step()

            if lr_scheduler is not None:
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
                os.path.join(config.save_ckpt_path, f"maskrcnn_epoch_{epoch}.pt"),
            )

        with torch.no_grad():
            print(f"Validation loop epoch {epoch}")
            # Validation steps
            map_metric = MeanAveragePrecision(
                iou_type="segm", class_metrics=True, extended_summary=True
            )
            map_metric_single_class = MeanAveragePrecision(
                iou_type="segm", class_metrics=True
            )
            # TODO: extended summary include IoU computation taking for each ground truth the argmax over the predictions.

            model.eval()

            img_to_label_dict = {}
            counter = 0

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
                for idx, outputs in enumerate(out):

                    indices = torch.where(outputs["scores"] > config.threshold_score)
                    labels = outputs["labels"][indices]
                    masks = (outputs["masks"].squeeze(1)[indices] > 0.5).type(
                        torch.bool
                    )

                    u = targets[idx].copy()
                    u["labels"] = torch.zeros(u["labels"].shape, dtype=torch.int)

                    map_metric.update(
                        [
                            {
                                "labels": labels,
                                "masks": masks,
                                "scores": outputs["scores"][indices].detach(),
                            }
                        ],
                        [targets[idx]],
                    )
                    map_metric_single_class.update(
                        [
                            {
                                "labels": torch.zeros(labels.shape, dtype=torch.int),
                                "masks": masks,
                                "scores": outputs["scores"][indices].detach(),
                            }
                        ],
                        [u],
                    )
                    # keep a mapping of img index and labels
                    img_to_label_dict[counter] = list(
                        targets[idx]["labels"].cpu().numpy()
                    )
                    counter += 1

                if epoch % config.log_images_every_epochs == 0:
                    if (
                        config.comet and i < 4
                    ):  # don't log all the images, it's too much
                        for j in range(len(targets)):

                            fig1 = get_ground_truth_viz(
                                torch.Tensor(im[j] * 255).type(torch.uint8)[:3, :, :],
                                targets[j],
                                categories,
                                cmap,
                            )
                            fig2 = get_preds_viz(
                                torch.Tensor(im[j] * 255)[:3, :, :].type(torch.uint8),
                                out[j],
                                config.threshold_score,
                                categories,
                                cmap,
                            )
                            fig = torch.cat((fig1, fig2), -1)
                            image_log = F.to_pil_image(fig)
                            experiment.log_image(
                                image_log, f"epoch_{epoch}_batch_{i}_img_{j}"
                            )

            map_score = map_metric.compute()
            map_single_score = map_metric_single_class.compute()

            if "ious" in map_score:
                ious = map_score["ious"]
                # ious: torchmetrics mAP extended summary ious which is a dictionary containing the IoU values for every image/class combination e.g. ious[(0,0)] would contain the IoU for image 0 and class 0. Each value is a tensor with shape (n,m) where n is the number of detections and m is the number of ground truth boxes for that image/class combination."""

                total_iou = 0
                num_instances = 0
                for elem in img_to_label_dict:
                    lab = img_to_label_dict[elem]
                    for label in np.unique(lab):
                        iou = ious[(elem, label)]
                        if iou == []:
                            # no instance was predicted for the class, IoU is 0 for every gt instance of that class
                            num_instances += lab.count(label)
                        else:

                            values, _ = iou.max(axis=0)
                            try:
                                num_instances += len(values)

                                total_iou += values.sum()
                            except:
                                num_instances += 1  # there is just one instance
                                total_iou += values.item()
                if config.comet:
                    experiment.log_metric(
                        "mIoU val single class trees", total_iou / num_instances
                    )
            else:
                print("map_score keys are", map_score.keys())

            if map_score["map"].item() > val_score:
                print("MAP val", map_score["map"].item())
                print(f"saving current best model at epoch {epoch}")
                val_score = map_score["map"].item()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_training_loss,
                    },
                    os.path.join(config.save_ckpt_path, "best_model.pt"),
                )

            if config.comet:
                experiment.log_metric("train_loss", epoch_training_loss)
                experiment.log_metric("mAP val", map_score["map"])
                experiment.log_metric("mAP single class val", map_single_score["map"])
            del map_metric
            del map_metric_single_class
            del img_to_label_dict
            del map_score
            # LOG MAP PER CLASS map_per_class

    # save last epoch model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_training_loss,
        },
        os.path.join(config.save_ckpt_path, f"maskrcnn_epoch_{epoch}.pt"),
    )

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
