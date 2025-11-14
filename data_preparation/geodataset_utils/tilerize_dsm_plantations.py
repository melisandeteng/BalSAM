from pathlib import Path

from geodataset.aoi import AOIFromPackageConfig
from geodataset.tilerize import RasterTilerizer

aoi_gpkg_config = AOIFromPackageConfig(
    aois={
        "train": Path("/network/projects/trees-co2/final_aois/merge_train_aois.gpkg"),
        "val": Path("/network/projects/trees-co2/final_aois/merge_val_aois.gpkg"),
        "test": Path("/network/projects/trees-co2/final_aois/merge_test_aois.gpkg"),
    }
)

rasters = [
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230608_cbpapinas_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230605_cbblackburn1_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230607_cbblackburn2_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230606_cbblackburn3_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230606_cbblackburn4_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230606_cbblackburn5_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230606_cbblackburn6_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230608_cbbernard1_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230608_cbbernard2_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230608_cbbernard3_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230608_cbbernard4_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230712_afcagauthier_itrf20_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230712_afcagauthmelpin_itrf20_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230712_afcahoule_itrf20_p1_dsm_highdis.cog.tif"
    ),
    Path(
        "/network/projects/trees-co2/dsm_orthos_aligned_nearest_take2/20230712_afcamoisan_itrf20_p1_dsm_highdis.cog.tif"
    ),
]

for i in range(len(rasters)):
    print(rasters[i])

    tilerizer = RasterTilerizer(
        raster_path=rasters[i],
        output_path=Path("/network/projects/trees-co2/final_dsm_tiles_v2"),
        tile_size=1024,
        tile_overlap=0.5,
        aois_config=aoi_gpkg_config,
        ground_resolution=None,  # optional, scale_factor must be None if used.
        scale_factor=None,  # optional, ground_resolution must be None if used
        ignore_black_white_alpha_tiles_threshold=0.8,  # optional
    )

    tilerizer.generate_tiles()
