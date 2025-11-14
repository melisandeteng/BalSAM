import json
from pathlib import Path

import albumentations
import cv2
import numpy as np
import rasterio
import tifffile
import torch
from geodataset.dataset.raster_dataset import \
    SegmentationLabeledRasterCocoDataset
from geodataset.utils import \
    coco_rle_segmentation_to_bbox as rle_segmentation_to_bbox
from pycocotools import mask as mask_utils
from shapely import MultiPolygon, Polygon
from torchvision import tv_tensors
from transforms import normalize_dsm


def polygon_to_mask(polygon, tile_height, tile_width):
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
    ):
        super().__init__(fold=fold, root_path=root_path, transform=transform)

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying any specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its annotations.
        """

        tile_info = self.tiles[idx]

        with rasterio.open(tile_info["path"]) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands
            # tile = np.moveaxis(tile, 0, -1)  # Channels last

        labels = tile_info["labels"]

        bboxes = []
        masks = []
        classes = []
        area = []

        for label in labels:

            segmentation = label["segmentation"]
            if ("is_rle_format" in label and label["is_rle_format"]) or isinstance(
                segmentation, dict
            ):
                # RLE format
                bbox = rle_segmentation_to_bbox(segmentation)
                mask = mask_utils.decode(segmentation)

            # elif (
            #    "is_rle_format" in label and not label["is_rle_format"]
            # ) or isinstance(segmentation, list):
            # Polygon (coordinates) format
            #    bbox = polygon_segmentation_to_bbox(segmentation)
            #    mask = polygon_to_mask(segmentation)
            # suppose all instances are not crowd

            else:
                raise NotImplementedError(
                    "Could not find the segmentation type (RLE vs polygon coordinates)."
                )

            masks.append(mask)
            bboxes.append([int(x) for x in bbox.bounds])
            classes.append(label["category_id"])
            area.append(
                (bbox.bounds[3] - bbox.bounds[1]) * (bbox.bounds[2] - bbox.bounds[0])
            )

        # Revisit normalization (normalisation de ImageNet ?)

        tile = tile / 255  # tv_tensors.Image(tile/255,  dtype=torch.float)

        category_ids = np.array(classes)

        if self.transform:
            transformed = self.transform(
                image=tile.transpose((1, 2, 0)),
                masks=masks,
                bboxes=bboxes,
                class_labels=category_ids,
            )
            transformed_image = transformed["image"].transpose((2, 0, 1))
            transformed_masks = transformed["masks"]
            transformed_bboxes = transformed["bboxes"]
            transformed_category_ids = transformed["class_labels"]

        else:
            transformed_image = tile
            transformed_masks = masks
            transformed_bboxes = bboxes
            transformed_category_ids = category_ids

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                transformed_bboxes, format="XYXY", canvas_size=tile.size
            ),
            "masks": torch.Tensor(np.array(transformed_masks)).type(torch.uint8),
            "labels": torch.Tensor(transformed_category_ids).type(torch.int64),
            "area": torch.FloatTensor(area),
            "iscrowd": torch.zeros(np.array(masks).shape[0]).type(torch.int64),
            "image_id": tile_info["path"],
        }
        return tv_tensors.Image(transformed_image, dtype=torch.float), target

    def __len__(self):
        return len(self.tiles)


class TreesDSMDataset:
    def __init__(
        self,
        json_file,
        dsm_normalization="gradient",
        transform: albumentations.core.composition.Compose = None,
    ):
        super().__init__()

        with open(json_file, "rb") as f:
            self.data = json.load(f)
        self.dsm_normalization = dsm_normalization
        self.transform = transform

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying any specified transformations.

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

        labels = data_im[
            "annotations"
        ]  # [d for d in self.data["annotations"] if d.get('image_id') ==data_im["id"]]

        bboxes = []
        masks = []
        classes = []
        area = []

        for label in labels:

            segmentation = label["segmentation"]
            if ("is_rle_format" in label and label["is_rle_format"]) or isinstance(
                segmentation, dict
            ):
                # RLE format
                bbox = rle_segmentation_to_bbox(segmentation)
                mask = mask_utils.decode(segmentation)

            # elif (
            #    "is_rle_format" in label and not label["is_rle_format"]
            # ) or isinstance(segmentation, list):
            # Polygon (coordinates) format
            #    bbox = polygon_segmentation_to_bbox(segmentation)
            #    mask = polygon_to_mask(segmentation, tile_h, tile_w)
            # suppose all instances are not crowd

            else:
                raise NotImplementedError(
                    "Could not find the segmentation type (RLE vs polygon coordinates)."
                )

            masks.append(mask)
            bboxes.append([int(x) for x in bbox.bounds])
            classes.append(label["category_id"])
            area.append(
                (bbox.bounds[3] - bbox.bounds[1]) * (bbox.bounds[2] - bbox.bounds[0])
            )

        # Revisit normalization (normalisation de ImageNet ?)
        # tile = tv_tensors.Image(tile/255,  dtype=torch.float)

        tile = tile / 255

        dsm = tifffile.imread(dsm_path)
        dsm = normalize_dsm(dsm, mode=self.dsm_normalization)
        dsm = np.expand_dims(dsm, axis=0)
        # dsm = tv_tensors.Image(dsm, dtype=torch.float)

        tile = np.concatenate((tile, dsm), axis=0)
        category_ids = np.array(classes)

        # TODO CHange transform to accomodate for img+dsm
        if self.transform:
            transformed = self.transform(
                image=tile.transpose((1, 2, 0)),
                masks=masks,
                bboxes=bboxes,
                class_labels=category_ids,
            )
            transformed_image = transformed["image"].transpose((2, 0, 1))
            transformed_masks = transformed["masks"]
            transformed_bboxes = transformed["bboxes"]
            transformed_category_ids = transformed["class_labels"]

        else:
            transformed_image = tile
            transformed_masks = masks
            transformed_bboxes = bboxes

        transformed_img = tv_tensors.Image(
            transformed_image, dtype=torch.float
        )  # torch.Tensor(tile, dtype= torch.float)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                transformed_bboxes, format="XYXY", canvas_size=tile.size
            ),
            "masks": torch.Tensor(np.array(transformed_masks)).type(torch.uint8),
            "labels": torch.Tensor(classes).type(torch.int64),
            "area": torch.FloatTensor(area),
            "iscrowd": torch.zeros(np.array(masks).shape[0]).type(torch.int64),
            "image_id": tile_path,  # label["image_id"],
        }
        return transformed_img, target

    def __len__(self):
        return len(self.data["images"])


class TreesDSMBCIDataset(SegmentationLabeledRasterCocoDataset):
    def __init__(
        self,
        fold: str,
        root_path: Path,
        transform: albumentations.core.composition.Compose = None,
        dsm_normalization="max",
    ):
        super().__init__(fold=fold, root_path=root_path, transform=transform)
        self.dsm_normalization = dsm_normalization

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
            tile = tile_file.read([1, 2, 3, 4]).astype(
                np.float32
            )  # Reading the first three bands
            # tile = np.moveaxis(tile, 0, -1)  # Channels last

        labels = tile_info["labels"]

        bboxes = []
        masks = []
        classes = []
        area = []

        for label in labels:

            segmentation = label["segmentation"]
            if ("is_rle_format" in label and label["is_rle_format"]) or isinstance(
                segmentation, dict
            ):
                # RLE format
                bbox = rle_segmentation_to_bbox(segmentation)
                mask = mask_utils.decode(segmentation)

            # elif (
            #    "is_rle_format" in label and not label["is_rle_format"]
            # ) or isinstance(segmentation, list):
            # Polygon (coordinates) format
            #    bbox = polygon_segmentation_to_bbox(segmentation)
            #    mask = polygon_to_mask(segmentation)
            # suppose all instances are not crowd

            else:
                raise NotImplementedError(
                    "Could not find the segmentation type (RLE vs polygon coordinates)."
                )

            masks.append(mask)
            bboxes.append([int(x) for x in bbox.bounds])
            classes.append(label["category_id"])
            area.append(
                (bbox.bounds[3] - bbox.bounds[1]) * (bbox.bounds[2] - bbox.bounds[0])
            )

        # Revisit normalization (normalisation de ImageNet ?)
        tile[:3, :, :] = (
            tile[:3, :, :] / 255
        )  # tv_tensors.Image(tile/255,  dtype=torch.float)
        tile[3, :, :] = normalize_dsm(tile[3, :, :], mode=self.dsm_normalization)

        category_ids = np.array(classes)

        if self.transform:
            transformed = self.transform(
                image=tile.transpose((1, 2, 0)),
                masks=masks,
                bboxes=bboxes,
                class_labels=category_ids,
            )

            transformed_image = transformed["image"].transpose((2, 0, 1))
            transformed_masks = transformed["masks"]
            transformed_bboxes = transformed["bboxes"]
            transformed_category_ids = transformed["class_labels"]

        else:
            transformed_image = tile
            transformed_masks = masks
            transformed_bboxes = bboxes
            transformed_category_ids = category_ids

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                transformed_bboxes, format="XYXY", canvas_size=tile.size
            ),
            "masks": torch.Tensor(np.array(transformed_masks)).type(torch.uint8),
            "labels": torch.Tensor(transformed_category_ids).type(torch.int64),
            "area": torch.FloatTensor(area),
            "iscrowd": torch.zeros(np.array(masks).shape[0]).type(torch.int64),
            "image_id": tile_info["path"],
        }
        return tv_tensors.Image(transformed_image, dtype=torch.float), target

    def __len__(self):
        return len(self.tiles)
