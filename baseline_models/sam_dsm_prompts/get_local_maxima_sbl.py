import json
import os

import numpy as np
import scipy
import tifffile
from skimage.feature import peak_local_max
from skimage.measure import label
from tqdm import tqdm


def get_local_maxima_from_folder(dsm_path, save_path):
    """
    paths: paths to dsm tiles
    save_path:
    """
    paths = os.listdir(dsm_path)

    for dsm_tile_path in tqdm(paths):
        dsf = tifffile.imread(os.path.join(dsm_path, dsm_tile_path))
        max_filtered = scipy.ndimage.maximum_filter(dsf, 100)
        # Find local maxima
        local_maxima = dsf == max_filtered

        # Print the coordinates of local maxima

        coords = np.argwhere(local_maxima)
        binary_image = np.zeros(dsf.shape)

        for coord in coords:
            binary_image[coord[0], coord[1]] = True

        # Perform connected component labeling
        labeled_image, num_labels = label(binary_image, connectivity=2, return_num=True)
        # Keep only one coordinate per group of adjacent pixels
        unique_coords = []
        for label_value in range(1, num_labels + 1):
            # Find coordinates belonging to the current label
            label_coords = np.argwhere(labeled_image == label_value)
            # Choose only one coordinate for the group
            centroid_coord = tuple(np.mean(label_coords, axis=0).astype(int))
            unique_coords.append(centroid_coord)
        np.save(
            os.path.join(
                save_path, os.path.basename(dsm_tile_path).strip(".tif") + ".npy"
            ),
            np.array(unique_coords),
        )


def get_local_maxima_from_file(dsm_path, save_path):
    """
    paths: paths to tif dsm tile
    save_path: .npy file path
    """

    dsf = tifffile.imread(os.path.join(dsm_path))

    # Print the coordinates of local maxima

    coordinates = peak_local_max(dsf, min_distance=20)

    binary_image = np.zeros(dsf.shape)

    for coord in coordinates:
        binary_image[coord[0], coord[1]] = True

    # Perform connected component labeling
    labeled_image, num_labels = label(binary_image, connectivity=2, return_num=True)
    # Keep only one coordinate per group of adjacent pixels
    unique_coords = []
    for label_value in range(1, num_labels + 1):
        # Find coordinates belonging to the current label
        label_coords = np.argwhere(labeled_image == label_value)
        # Choose only one coordinate for the group
        centroid_coord = tuple(np.mean(label_coords, axis=0).astype(int))
        unique_coords.append(centroid_coord)
        np.save(save_path, np.array(unique_coords))


if __name__ == "__main__":

    with open(
        "/network/projects/trees-co2/quebec_trees_tiles_fullresolution/merged_annots_dsm_test_new.json",
        "rb",
    ) as f:
        annots = json.load(f)
    tiles = annots["images"]
    save_path = (
        "/network/projects/trees-co2/experiments/segmate_height_prompts_sbli_mindist20"
    )
    for t in tiles:
        dsm_path = t["dsm_path"]
        name_img = os.path.basename(t["file_name"]).strip(".tif") + ".npy"

        get_local_maxima_from_file(dsm_path, os.path.join(save_path, name_img))
    print("done")
