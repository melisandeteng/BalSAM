import json
from pathlib import Path

from geodataset.aoi import AOIFromPackageConfig
from geodataset.tilerize import LabeledRasterTilerizer

with open("../geodataset_utils/categories_subset_family.json", "r") as f:
    CATEGORIES = json.load(f)["categories"]


aoi_gpkg_config = AOIFromPackageConfig(
    aois={
        "train": Path(
            "/network/projects/trees-co2/BCI/BCI_aois/20220929_aoi_train.gpkg"
        ),
        "val": Path("/network/projects/trees-co2/BCI/BCI_aois/20220929_aoi_valid.gpkg"),
        "test": Path("/network/projects/trees-co2/BCI/BCI_aois/20220929_aoi_test.gpkg"),
    }
)

rasters = [
    Path(
        "/network/projects/trees-co2/BCI/BCI_50ha_2022_09_29_crownmap_raw/BCI_50ha_2022_09_29_global.tif"
    )
]
labels = [
    Path(
        "/network/projects/trees-co2/BCI/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved_clean_family.shp"
    )
]

for i in range(len(rasters)):
    print(rasters[i])

    tilerizer = LabeledRasterTilerizer(
        raster_path=rasters[i],
        labels_path=labels[i],
        output_path=Path(
            "/network/projects/trees-co2/BCI/BCI_2022_tilessubset_family_full"
        ),
        tile_size=1024,
        tile_overlap=0.5,
        aois_config=aoi_gpkg_config,
        # geopackage_layer_name=geopackage_layer_names[i],
        ground_resolution=None,  # optional, scale_factor must be None if used.
        scale_factor=None,  # optional, ground_resolution must be None if used.
        use_rle_for_labels=True,  # optional
        min_intersection_ratio=0.1,  # optional
        ignore_tiles_without_labels=True,  # optional
        ignore_black_white_alpha_tiles_threshold=0.8,  # optional
        main_label_category_column_name="family",  # optional
        other_labels_attributes_column_names=[],  # optional
        coco_categories_list=CATEGORIES,
    )

    tilerizer.generate_coco_dataset()
