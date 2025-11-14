from pathlib import Path

from geodataset.aoi import AOIFromPackageConfig
from geodataset.tilerize import LabeledRasterTilerizer

CATEGORIES = [
    {
        "id": 1,
        "name": "dead",
        "global_id": 1.0,
        "is_taxon": False,
        "rank": None,
        "other_names": ["Mort"],
        "supercategory": None,
    },
    {
        "id": 2,
        "name": "Pinopsida",
        "global_id": 194,
        "is_taxon": True,
        "rank": "class",
        "other_names": ["Conifere"],
        "supercategory": 7707728,
    },
    {
        "id": 3,
        "name": "Magnoliopsida",
        "global_id": 220,
        "is_taxon": True,
        "rank": "class",
        "other_names": ["Feuillus", "QURU", "OSVI", "PRPE", "FRNI"],
        "including_names": [
            "Quercus rubra L.",
            "Ostrya virginiana (Mill.) K.Koch",
            "Prunus pensylvanica L.fil.",
            "Fraxinus nigra Marshall",
        ],
        "including_global_ids": [2880539, 5332289, 8202030, 3172369],
        "supercategory": 7707728,
    },
    {
        "id": 4,
        "name": "Thuja occidentalis L.",
        "global_id": 2684178,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Thuja occidentalis", "THOC"],
        "supercategory": 2684168,
    },
    {
        "id": 5,
        "name": "Abies balsamea (L.) Mill.",
        "global_id": 2685383,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Abies balsamea", "ABBA"],
        "supercategory": 2684876,
    },
    {
        "id": 6,
        "name": "Larix laricina (Du Roi) K.Koch",
        "global_id": 2686231,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Larix laricina", "LALA"],
        "supercategory": 2686156,
    },
    {
        "id": 7,
        "name": "Tsuga canadensis (L.)",
        "global_id": 2687182,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Tsuga canadensis", "TSCA"],
        "supercategory": 8527396,
    },
    {
        "id": 8,
        "name": "Betula L.",
        "global_id": 2875008,
        "is_taxon": True,
        "rank": "genus",
        "other_names": ["Betula", "BEPO"],
        "including_names": ["Betula populifolia Marshall"],
        "including_global_ids": [8184083],
        "supercategory": 4688,
    },
    {
        "id": 9,
        "name": "Fagus grandifolia Ehrh.",
        "global_id": 2882274,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Fagus grandifolia", "FAGR"],
        "supercategory": 2874875,
    },
    {
        "id": 10,
        "name": "Populus L.",
        "global_id": 3040183,
        "is_taxon": True,
        "rank": "genus",
        "other_names": ["Populus", "POBA", "POGR", "POTR"],
        "supercategory": 6664,
        "including_names": [
            "Populus balsamifera L.",
            "Populus grandidentata Michx.",
            "Populus tremuloides Michx.",
        ],
        "including_global_ids": [3040184, 3040210, 3040215],
    },
    {
        "id": 11,
        "name": "Acer L.",
        "global_id": 3189834,
        "is_taxon": True,
        "rank": "genus",
        "other_names": ["Acer"],
        "supercategory": 6657,
    },
    {
        "id": 12,
        "name": "Acer pensylvanicum L.",
        "global_id": 3189836,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Acer pensylvanicum", "ACPE"],
        "supercategory": 3189834,
    },
    {
        "id": 13,
        "name": "Acer saccharum Marshall",
        "global_id": 3189859,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Acer saccharum", "ACSA"],
        "supercategory": 3189834,
    },
    {
        "id": 14,
        "name": "Acer rubrum L.",
        "global_id": 3189883,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Acer rubrum", "ACRU"],
        "supercategory": 3189834,
    },
    {
        "id": 15,
        "name": "Pinus strobus L.",
        "global_id": 5284982,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Pinus strobus", "PIST"],
        "supercategory": 2684241,
    },
    {
        "id": 16,
        "name": "Betula alleghaniensis Britton",
        "global_id": 5331779,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Betula alleghaniensis", "BEAL"],
        "supercategory": 2875008,
    },
    {
        "id": 17,
        "name": "Betula papyrifera Marshall",
        "global_id": 5332120,
        "is_taxon": True,
        "rank": "species",
        "other_names": ["Betula papyrifera", "BEPA"],
        "supercategory": 2875008,
    },
    {
        "id": 18,
        "name": "Picea A.Dietr.",
        "global_id": 7606064,
        "is_taxon": True,
        "rank": "genus",
        "other_names": ["Picea", "PIGL", "PIMA", "PIRU"],
        "supercategory": 3925,
        "including_names": [
            "Picea rubens Sarg.",
            "Picea glauca (Moench) Voss",
            "Picea mariana (Mill.) Britton et al.",
        ],
        "including_global_ids": [5284717, 5284745, 5284802],
    },
]


aoi_gpkg_config = AOIFromPackageConfig(
    aois={
        "train": Path(
            "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/train_aoi.geojson"
        ),
        "val": Path(
            "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/valid_aoi.geojson"
        ),
        "test": Path(
            "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/inference_zone.gpkg"
        ),
    }
)

rasters = [
    Path(
        "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/2021-09-02/zone2/2021-09-02-sbl-z2-rgb-cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif"
    ),
]
labels = [
    Path(
        "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg"
    ),
    Path(
        "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/Z2_polygons.gpkg"
    ),
    Path(
        "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/Z3_polygons.gpkg"
    ),
]
for i in range(len(rasters)):
    print(rasters[i])

    # scale_factor=None for full res, 0.8 for highres, 0.5 for halfres
    tilerizer = LabeledRasterTilerizer(
        raster_path=rasters[i],
        labels_path=labels[i],
        output_path=Path("/network/projects/trees-co2/quebec_trees_tiles_highres"),
        tile_size=1024,
        tile_overlap=0.5,
        aois_config=aoi_gpkg_config,
        # geopackage_layer_name=geopackage_layer_names[i],
        ground_resolution=None,  # optional, scale_factor must be None if used.
        scale_factor=0.8,  # optional, ground_resolution must be None if used.
        use_rle_for_labels=True,  # optional
        min_intersection_ratio=0.2,  # optional
        ignore_tiles_without_labels=True,  # optional
        ignore_black_white_alpha_tiles_threshold=0.8,  # optional
        main_label_category_column_name="Label",  # optional
        other_labels_attributes_column_names=[],  # optional
        coco_categories_list=CATEGORIES,
    )

    tilerizer.generate_coco_dataset()
