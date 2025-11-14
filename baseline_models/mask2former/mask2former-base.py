from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from transformers import Mask2FormerConfig, Mask2FormerModel
import json
import os
from pathlib import Path
from PIL import Image
import albumentations
import numpy as np
import rasterio
import tifffile
from pycocotools import mask as mask_utils
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from geodataset.dataset.raster_dataset import \
    SegmentationLabeledRasterCocoDataset
from geodataset.utils import \
    coco_rle_segmentation_to_bbox as rle_segmentation_to_bbox
from torch.utils.data import DataLoader
import albumentations as a
import comet_ml
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torchvision
from torch import Tensor, nn
from torchvision import ops
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from transformers import TrainingArguments, Trainer
# Before setting, it might be using an old key or none
# if you haven't logged in or set an environment variable externally.
SEED = 4021
torch.manual_seed(SEED)
# Set the environment variable
os.environ["COMET_API_KEY"] = YOUR_COMET_API_KEY
os.environ["COMET_WORKSPACE"]= YOUR_COMET_WORKSPACE

# Choose a pre-trained Mask2Former model
# For instance segmentation, models pre-trained on COCO are good starting points.
model_name = "facebook/mask2former-swin-base-coco-instance"
# or "facebook/mask2former-swin-base-coco-instance" for a smaller model

processor = Mask2FormerImageProcessor.from_pretrained(model_name, num_labels =9, do_reduce_labels=True)
# Initializing a Mask2Former facebook/mask2former-swin-small-coco-instance configuration
config= Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-coco-instance")
#config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-coco")

# Your custom dataset might have a different number of classes
# Example: If your dataset has 5 foreground classes (and Mask2Former adds one for "no object")
config.num_labels = 9
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True )


def normalize_dsm(dsm, mode="max", means_dsm=None, stds_dsm=None):
    """
    dsm: numpy array of the dsm
    mode: "minmax", "max", "gradient"
    """
    if mode == "max":
        return dsm / dsm.max()
    elif mode == "minmax":
        return dsm - dsm.min() / dsm.max() - dsm.min()
    elif mode == "gradient":
        return np.gradient(dsm, axis=-1)
    elif mode == "all_gradients": 
        means_dsm = means_dsm.reshape(2, 1, 1)
        stds_dsm = stds_dsm.reshape(2, 1, 1)
        return (dsm - means_dsm) / stds_dsm
    else:
        raise ValueError("mode of normalization not supported")


        

def polygon_to_mask(polygon):
    binary_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

    # Function to process each polygon
    def process_polygon(p):
        contours = np.array(p.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(binary_mask, [contours], 1)

    if isinstance(polygon, Polygon):
        process_polygon(polygon)
    elif isinstance(polygon, MultiPolygon):
        for polygon in polygon.geoms:
            process_polygon(polygon)
    else:
        raise TypeError("Geometry must be a Polygon or MultiPolygon")

    return binary_mask


class TreesDataset(SegmentationLabeledRasterCocoDataset):
    def __init__(
        self,
        fold: str,
        root_path: Path,
        transform: albumentations.core.composition.Compose = None,
        processor=None
    ):
        super().__init__(fold=fold, root_path=root_path, transform=transform)
        self.processor = processor
        
    def __getitem__(self, idx: int):
        """Retrieves a tile and its annotations by index, applying any
        specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its annotations.
        """

        tile_info = self.tiles[idx]

        with rasterio.open(tile_info["path"]) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands
            # tile = np.moveaxis(tile, 0, -1)  # Channels last

        targets = {}

        labels = tile_info["labels"]

        bboxes = []
        masks = []
        classes = {}
        area = []

        for k, label in enumerate(labels):

            segmentation = label["segmentation"]
            if ("is_rle_format" in label and label["is_rle_format"]) or isinstance(
                segmentation, dict
            ):
                # RLE format
                bbox = rle_segmentation_to_bbox(segmentation)
                mask = mask_utils.decode(segmentation)

            elif (
                "is_rle_format" in label and not label["is_rle_format"]
            ) or isinstance(segmentation, list):
                # Polygon (coordinates) format
                bbox = polygon_segmentation_to_bbox(segmentation)
                mask = polygon_to_mask(segmentation)
                # suppose all instances are not crowd

            else:
                raise NotImplementedError(
                    "Could not find the segmentation type (RLE vs polygon coordinates)."
                )

            bboxes.append([int(x) for x in bbox.bounds])
            classes[k+1]=label["category_id"]
            area.append(
                (bbox.bounds[3] - bbox.bounds[1]) * (bbox.bounds[2] - bbox.bounds[0])
            )
            masks.append(mask*(k+1))
        masks = np.max(masks, axis = 0)
        
        inputs = processor(
            images=[tile],
            segmentation_maps=[masks], # Pass the list of binary masks
            instance_id_to_semantic_id=[classes], # Pass the list of class labels for each instance
            task_inputs="instance_segmentation",
            return_tensors="pt"
        )
        
        #print(inputs["mask_labels"][0].shape)
        #for k, v in inputs.items():
        #    if isinstance(v, torch.Tensor):
        #        inputs[k] = v.squeeze(0)
        #    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
        #        inputs[k] = [item.squeeze(0) for item in v]
        return(inputs)

    def __len__(self):
        return len(self.tiles)

root_path = "/network/projects/trees-co2/final_tiles/"
transform = a.Compose(
            [a.HorizontalFlip(p=0.5), a.VerticalFlip(p=0.5)],
            bbox_params=a.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )
 
train_dataset = TreesDataset("train", 
        root_path=root_path,
        transform=transform, 
        processor=processor
    )
test_dataset = TreesDataset("val", 
        root_path=root_path,
        transform=None, 
        processor=processor
    )




# Define your training arguments
training_args = TrainingArguments(
    output_dir=f"/network/scratch/t/tengmeli/mask2former_fine_tuned_balsam/baseswin_{SEED}",
    per_device_train_batch_size=8, # Adjust based on your GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=100, # Or more, depending on dataset size and convergence
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"/network/scratch/t/tengmeli/mask2former_fine_tuned_balsam/logs_baseswin_{SEED}",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    #evaluation_strategy="steps",
    eval_steps=1000,
    remove_unused_columns=False, # Important for custom datasets
    fp16=True, # Enable mixed precision training if your GPU supports it
    dataloader_num_workers=4, # Number of workers for data loading
)


custom_id2label = {
    #0: "background",
    0: "piba",
    1: "pima",
    2: "pist",
    3: "pigl",
    4:"thoc",
    5: "ulam",
    6:"other",
    7:"beal",
    8:"acsa"
}


# 3. Update the model's configuration
# This is the crucial step. You update the `id2label` attribute in the model's config.
model.config.backbone_config.id2label = custom_id2label

# 4. Update the number of labels (if your dataset has a different number of classes)
# The `num_labels` attribute in the config controls the output dimension of the classification head.
# If your custom dataset has a different number of classes than the pre-trained model,
# you MUST update this to match len(custom_id2label).
model.config.num_labels = len(custom_id2label)

# Important: If you change num_labels, the classification head's weights will be randomly
# re-initialized for the new classes (or potentially adapted, but usually re-initialized).
# This means you'll need to train this new head from scratch.

# You can also update the `label2id` if you need that for convenience
model.config.backbone_config.label2id = {v: k for k, v in custom_id2label.items()}

def data_collator(features):
    
    images = torch.vstack([feature['pixel_values'] for feature in features])
    im_mask = torch.vstack([feature['pixel_mask'] for feature in features])

    class_labels = [feature['class_labels'][0].type(torch.LongTensor) for feature in features] #[[feature['class_labels'][0].type(torch.LongTensor) if len(feature['class_labels'][0])==1 else for feature in features] 
    
    #[[feature['class_labels'][0].type(torch.LongTensor)] if len(feature['class_labels'][0])==1 else [torch.LongTensor(f) for f in feature['class_labels'][0]] for feature in features]
    
    # Make sure your dataset's __getitem__ returns these keys
    # For instance, `mask_labels` and `class_labels` should already be lists of tensors
    # or have been processed into that format.

    # The image processor handles the heavy lifting:
    # - Resizing/normalizing images to pixel_values
    # - Padding images if necessary (creates pixel_mask)
    # - Preparing mask_labels and class_labels for the model's input
    #   (e.g., converting to Set-level targets, padding masks, etc.)
    batch = {'pixel_values': images, "pixel_mask": im_mask, 'mask_labels': [feature['mask_labels'][0].type(torch.FloatTensor) for feature in features],  
             "class_labels": class_labels}
    return(batch)


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    # You might want to define a custom compute_metrics function for evaluation
    # For instance segmentation, common metrics are mAP (mean Average Precision)
    # as defined in COCO. This usually requires `pycocotools`.
)

# Start training
trainer.train() #resume_from_checkpoint=True)

if __name__==main():
    print("training")
