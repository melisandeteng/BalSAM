from pathlib import Path

from geodataset.aoi import AOIFromPackageConfig
from geodataset.tilerize import LabeledRasterTilerizer

CATEGORIES = [
    {"id": 1, "name": "piba", "supercategory": "pinus"},
    {"id": 2, "name": "pima", "supercategory": "picea"},
    {"id": 3, "name": "pist", "supercategory": "pinus"},
    {"id": 4, "name": "pigl", "supercategory": "picea"},
    {"id": 5, "name": "thoc", "supercategory": "thuja"},
    {"id": 6, "name": "ulam", "supercategory": "ulmaceae"},
    {
        "id": 7,
        "name": "other",
        "other_names": ["qeru", "bepa", "potr", "lala", "pire"],
        "supercategory": "",
    },
    {"id": 8, "name": "beal", "supercategory": "betula"},
    {"id": 9, "name": "acsa", "supercategory": "acer"},
]


aoi_gpkg_config = AOIFromPackageConfig(
    aois={
        "train": Path("/network/projects/trees-co2/final_aois/merge_train_aois.gpkg"),
        "val": Path("/network/projects/trees-co2/final_aois/merge_val_aois.gpkg"),
        "test": Path("/network/projects/trees-co2/final_aois/merge_test_aois.gpkg"),
    }
)

geopackage_layer_names = [
    "20230608_cbpapinas_p1_labels_masks",
    "20230605_cbblackburn1_p1_labels_masks",
    "20230607_cbblackburn2_p1_labels_masks",
    "20230606_cbblackburn3_p1_labels_masks",
    "20230606_cbblackburn4_p1_labels_masks",
    "20230606_cbblackburn5_p1_labels_masks",
    "20230606_cbblackburn6_p1_labels_masks",
    "20230608_cbbernard1_p1_labels_masks",
    "20230608_cbbernard2_p1_labels_masks",
    "20230608_cbbernard3_p1_labels_masks",
    "20230608_cbbernard4_p1_labels_masks",
    "20230712_afcagauthier_itrf20_p1_labels_masks",
    "20230712_afcagauthmelpin_itrf20_p1_labels_masks",
    "20230712_afcahoule_itrf20_p1_labels_masks",
    "20230712_afcamoisan_itrf20_p1_labels_masks",
]
rasters = [
    Path("/network/projects/trees-co2/orthos/20230608_cbpapinas_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230605_cbblackburn1_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230607_cbblackburn2_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230606_cbblackburn3_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230606_cbblackburn4_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230606_cbblackburn5_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230606_cbblackburn6_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230608_cbbernard1_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230608_cbbernard2_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230608_cbbernard3_p1_rgb.cog.tif"),
    Path("/network/projects/trees-co2/orthos/20230608_cbbernard4_p1_rgb.cog.tif"),
    Path(
        "/network/projects/trees-co2/orthos/20230712_afcagauthier_itrf20_p1_rgb.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/orthos/20230712_afcagauthmelpin_itrf20_p1_rgb.cog.tif"
    ),
    Path("/network/projects/trees-co2/orthos/20230712_afcahoule_itrf20_p1_rgb.cog.tif"),
    Path(
        "/network/projects/trees-co2/orthos/20230712_afcamoisan_itrf20_p1_rgb.cog.tif"
    ),
]
labels = [
    Path("/network/projects/trees-co2/Donnees_finales/20230608_cbpapinas_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230605_cbblackburn1_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230607_cbblackburn2_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230606_cbblackburn3_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230606_cbblackburn4_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230606_cbblackburn5_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230606_cbblackburn6_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230608_cbbernard1_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230608_cbbernard2_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230608_cbbernard3_p1.gpkg"),
    Path("/network/projects/trees-co2/Donnees_finales/20230608_cbbernard4_p1.gpkg"),
    Path(
        "/network/projects/trees-co2/Donnees_finales/20230712_afcagauthier_itrf20_p1.gpkg"
    ),
    Path(
        "/network/projects/trees-co2/Donnees_finales/20230712_afcagauthmelpin_itrf20_p1.gpkg"
    ),
    Path(
        "/network/projects/trees-co2/Donnees_finales/20230712_afcahoule_itrf20_p1.gpkg"
    ),
    Path(
        "/network/projects/trees-co2/Donnees_finales/20230712_afcamoisan_itrf20_p1.gpkg"
    ),
]
for i in range(len(geopackage_layer_names)):
    print(rasters[i])

    tilerizer = LabeledRasterTilerizer(
        raster_path=rasters[i],
        labels_path=labels[i],
        output_path=Path("/network/projects/trees-co2/final_tiles"),
        tile_size=1024,
        tile_overlap=0.5,
        aois_config=aoi_gpkg_config,
        geopackage_layer_name=geopackage_layer_names[i],
        ground_resolution=None,  # optional, scale_factor must be None if used.
        scale_factor=None,  # optional, ground_resolution must be None if used.
        use_rle_for_labels=True,  # optional
        min_intersection_ratio=0.2,  # optional
        ignore_tiles_without_labels=True,  # optional
        ignore_black_white_alpha_tiles_threshold=0.8,  # optional
        main_label_category_column_name="class_code",  # optional
        other_labels_attributes_column_names=None,  # optional
        coco_categories_list=CATEGORIES,
    )

    tilerizer.generate_coco_dataset()
