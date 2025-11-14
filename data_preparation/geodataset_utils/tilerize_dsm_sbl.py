from pathlib import Path

from geodataset.aoi import AOIFromPackageConfig
from geodataset.tilerize import RasterTilerizer

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
        "/network/projects/trees-co2/quebec_trees/dsm_orthos_aligned/20210902_sblz1_p4rtk_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/quebec_trees/dsm_orthos_aligned/20210902_sblz2_p4rtk_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/quebec_trees/dsm_orthos_aligned/20210902_sblz3_p4rtk_dsm_highdis.cog.tif"
    ),
]

for i in range(len(rasters)):
    print(rasters[i])

    tilerizer = RasterTilerizer(
        raster_path=rasters[i],
        output_path=Path("/network/projects/trees-co2/quebec_trees_dsm_tiles_fullres"),
        tile_size=1024,
        tile_overlap=0.5,
        aois_config=aoi_gpkg_config,
        ground_resolution=None,  # optional, scale_factor must be None if used.
        scale_factor=None,  # optional, ground_resolution must be None if used
        ignore_black_white_alpha_tiles_threshold=0.8,  # optional
    )

    tilerizer.generate_tiles()
