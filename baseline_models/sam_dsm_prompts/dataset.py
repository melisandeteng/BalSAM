import base64
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
from geodataset.dataset.raster_dataset import \
    SegmentationLabeledRasterCocoDataset
from pycocotools import mask as mask_utils
from shapely import MultiPolygon, Polygon


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


class SAMTreesDataset(SegmentationLabeledRasterCocoDataset):
    def __init__(self, fold: str, root_path: Path, height_prompts_path: Path):
        super().__init__(fold=fold, root_path=root_path)

        self.height_prompts_path = height_prompts_path

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its annotations by index, applying any specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its annotations.
        """

        tile_info = self.tiles[idx]

        name_tile = os.path.basename(str(tile_info["path"])).split(".")[0]
        tile = tifffile.imread(str(tile_info["path"]))
        tile = tile[:, :, :3]
        tile = np.transpose(tile, (2, 1, 0))
        # with rasterio.open(tile_info['path']) as tile_file:
        #   tile = tile_file.read([1, 2, 3])  # Reading the first three bands
        # tile = np.moveaxis(tile, 0, -1)  # Channels last
        tile_h, tile_w = tile.shape[0], tile.shape[1]
        masks = []

        labels = tile_info["labels"]

        for label in labels:

            segmentation = label["segmentation"]
            if ("is_rle_format" in label and label["is_rle_format"]) or isinstance(
                segmentation, dict
            ):
                # RLE format
                try:
                    mask = mask_utils.decode(segmentation)
                except:
                    encoded_counts = segmentation["counts"]
                    decoded_counts = base64.b64decode(encoded_counts)
                    segmentation["counts"] = decoded_counts
                    mask = mask_utils.decode(segmentation)

            elif (
                "is_rle_format" in label and not label["is_rle_format"]
            ) or isinstance(segmentation, list):
                # Polygon (coordinates) format
                mask = polygon_to_mask(segmentation, tile_h, tile_w)
                # suppose all instances are not crowd

            else:
                raise NotImplementedError(
                    "Could not find the segmentation type (RLE vs polygon coordinates)."
                )

            masks.append(mask.T)

        target = {
            "masks": (np.array(masks) > 0).astype(np.uint8),
            "image_id": label["image_id"],
            "image_path": str(tile_info["path"]),
            "prompts": np.load(
                os.path.join(self.height_prompts_path, name_tile + ".npy")
            ),
        }
        return (tile, target)

    def __len__(self):
        return len(self.tiles)
