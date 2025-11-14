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

#SEED=0
#torch.manual_seed(0)

# Set the environment variable
os.environ["COMET_API_KEY"] = "JAQ6zQMoTH7snvbIkpjeBswPW"
os.environ["COMET_WORKSPACE"]="melisandeteng"

# if you haven't logged in or set an environment variable externally.
# Choose a pre-trained Mask2Former model
# For instance segmentation, models pre-trained on COCO are good starting points.
model_name = "facebook/mask2former-swin-base-coco-instance"
# or "facebook/mask2former-swin-base-coco-instance" for a smaller model

processor = Mask2FormerImageProcessor.from_pretrained(model_name, num_labels =9,do_reduce_labels=True)

# Initializing a Mask2Former facebook/mask2former-swin-small-coco-instance configuration
config= Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-coco-instance")
#config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-coco")

# Your custom dataset might have a different number of classes
# Example: If your dataset has 5 foreground classes (and Mask2Former adds one for "no object")
config.num_labels = 9

model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True )


original_first_conv = model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection

print(f"Original first conv layer: {original_first_conv}")
print(f"Original in_channels: {original_first_conv.in_channels}")

# 3. Create a new convolutional layer with 4 input channels
# It should have the same output channels, kernel size, stride, padding, and bias as the original.
new_in_channels = 4
new_first_conv = torch.nn.Conv2d(
    in_channels=new_in_channels,
    out_channels=original_first_conv.out_channels,
    kernel_size=original_first_conv.kernel_size,
    stride=original_first_conv.stride,
    padding=original_first_conv.padding,
    bias=original_first_conv.bias is not None
)

# 4. Initialize the weights of the new convolutional layer
# This is crucial. You have 3 channels from the pre-trained weights, and one new channel.
# Option A: Copy original weights for the first 3 channels and initialize the 4th randomly (e.g., to zeros or a small random value).
# This is generally the preferred method to retain pre-trained knowledge.
with torch.no_grad():
    # Copy weights for existing channels
    new_first_conv.weight[:, :original_first_conv.in_channels, :, :].copy_(
        original_first_conv.weight
    )
    # Initialize weights for the new channel (e.g., with zeros or a small random value)
    # This assumes the 4th channel is additional information.
    # If the 4th channel is meant to replace one of the others, adjust accordingly.
    new_first_conv.weight[:, original_first_conv.in_channels:, :, :].zero_() # Or use torch.nn.init.kaiming_normal_ etc.
    if original_first_conv.bias is not None:
        new_first_conv.bias.copy_(original_first_conv.bias)

# Option B: Reinitialize all weights randomly for the new layer.
# This might be necessary if your 4-channel data is vastly different from RGB,
# but you'll lose a lot of the pre-trained benefits.
# torch.nn.init.kaiming_normal_(new_first_conv.weight, mode='fan_out', nonlinearity='relu')
# if new_first_conv.bias is not None:
#     torch.nn.init.zeros_(new_first_conv.bias)


# 5. Replace the old convolutional layer with the new one
model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_first_conv

print(f"\nModified first conv layer: {model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection}")
print(f"New in_channels: {model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.in_channels}")



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


class TreesDSMDataset:
    def __init__(
        self,
        json_file,
        dsm_normalization="gradient",
        transform: albumentations.core.composition.Compose = None,
        means_dsm=None,
        stds_dsm=None,
    ):
        super().__init__()

        with open(json_file, "rb") as f:
            self.data = json.load(f)
        self.dsm_normalization = dsm_normalization
        self.transform = transform
        self.means_dsm = means_dsm
        self.stds_dsm = stds_dsm
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
    def __getitem__(self, idx: int):
        """Retrieves a tile and its annotations by index, applying any
        specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its annotations.
        """

        data_im = self.data["images"][idx]

        tile_path = data_im["file_name"]
        dsm_path = data_im["dsm_path"]

        with rasterio.open(tile_path) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands
            # tile = np.moveaxis(tile, 0, -1)  # Channels last

        targets = {}

        labels = data_im["annotations"]#[d for d in self.data["annotations"] if d.get('image_id') ==data_im["id"]]

        bboxes = []
        masks = []
        classes = {}
        area = []

        for k,label in enumerate(labels):

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

            masks.append(mask*(k+1))
            bboxes.append([int(x) for x in bbox.bounds])
            classes[k+1]=label["category_id"]
            area.append(
                (bbox.bounds[3] - bbox.bounds[1]) * (bbox.bounds[2] - bbox.bounds[0])
            )
        
        masks = np.max(masks, axis = 0)
        # Revisit normalization (normalisation de ImageNet ?)
        # tile = tv_tensors.Image(tile/255,  dtype=torch.float)

        #tile = tile / 255
        
        
       

        dsm = tifffile.imread(dsm_path)
        dsm = normalize_dsm(
            dsm,
            mode=self.dsm_normalization,
            means_dsm=self.means_dsm,
            stds_dsm=self.stds_dsm,
        )
        if self.dsm_normalization != "all_gradients":
            dsm = np.expand_dims(dsm, axis=0)
        # dsm = tv_tensors.Image(dsm, dtype=torch.float)

        #tile = np.concatenate((tile, dsm), axis=0)
        #category_ids = np.array(classes)

        tile = np.concatenate((tile, dsm), axis=0)
        category_ids = np.array(classes)


        # TODO CHange transform to accomodate for img+dsm
        if self.transform:
           
            transformed = self.transform(
                image=tile.transpose((1, 2, 0)),
                masks=np.expand_dims(masks, axis = 0),
                bboxes=bboxes,
                class_labels=list(classes.values()),
            )
            tile = transformed["image"].transpose((2, 0, 1))
            masks = transformed["masks"]
    
        dsm = tile[3,:,:]
        tile = tile[:3,:,:]
        # transformed_img = torch.concatenate((transformed_image, transformed_dsm))

        inputs = processor(
            images=[tile],
            segmentation_maps=[masks[0]], # Pass the list of binary masks
            instance_id_to_semantic_id=[classes], # Pass the list of class labels for each instance
            task_inputs="instance_segmentation",
            return_tensors="pt"
        )
 
        dsm = torch.nn.functional.interpolate(
                torch.from_numpy(dsm).unsqueeze(0).unsqueeze(0), # Add batch dim for interpolate
                size=(384, 384),
                mode='bilinear',
                align_corners=False
            ) # Remove batch dim
        
        pixel_values_4_channel  = torch.cat([inputs["pixel_values"], dsm], axis = 1)
        # Convert to PyTorch tensor and permute to (C, H, W)
        #pixel_values_4_channel = torch.zeros(4,384,384).float() # (4, H, W)

        # Apply normalization to each channel group (important!)
        # Scale RGB to 0-1, then normalize with mean/std
        #pixel_values_4_channel[:3] = pixel_values_4_channel[:3] 
        # Scale extra channel to 0-1 (if originally 0-255), then normalize with its own mean/std
        #pixel_values_4_channel[3] = dsm
        inputs["pixel_values"] =  pixel_values_4_channel
        return inputs
    def __len__(self):
        return len(self.data["images"])


root_path = "/network/projects/trees-co2/final_tiles/"

transform = a.Compose(
            [a.HorizontalFlip(p=0.5), a.VerticalFlip(p=0.5)],
            bbox_params=a.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )
train_dataset = TreesDSMDataset("/network/projects/trees-co2/final_tiles/merged_annots_dsm_train_new_maskrcnn.json", 
               dsm_normalization="max",
                     transform = transform
                     
       )
test_dataset = TreesDSMDataset("/network/projects/trees-co2/final_tiles/merged_annots_dsm_val_new_maskrcnn.json", 
               dsm_normalization="max",
                  
                     
       )



# Define your training arguments
training_args = TrainingArguments(
    output_dir=f"/network/scratch/t/tengmeli/mask2former_fine_tuned_dsm_balsam/resized_dsm",
    per_device_train_batch_size=8, # Adjust based on your GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=100, # Or more, depending on dataset size and convergence
    learning_rate=1e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir=f"/network/scratch/t/tengmeli/mask2former_fine_tuned_dsm_balsam/resized_dsm",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    #eval_strategy="epoch",
    #metric_for_best_model="mAP",  
    #greater_is_better=True, 
    eval_steps=1000,
    remove_unused_columns=False, # Important for custom datasets
    fp16=True, # Enable mixed precision training if your GPU supports it
    dataloader_num_workers=4, # Number of workers for data loading
)



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

custom_id2label = {
    
    0: "pinus",
    1: "picea",
    2: "pinus",
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
trainer.train()

if __name__==main():
    os.environ["COMET_API_KEY"] = "JAQ6zQMoTH7snvbIkpjeBswPW"
    os.environ["COMET_WORKSPACE"]="melisandeteng"

    print("training")