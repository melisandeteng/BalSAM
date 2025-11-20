# python

import os
import pathlib

import rasterio
from rasterio.warp import Resampling
from rasterio.windows import from_bounds

if __name__ == "__main__":

    PROJECT_DATA_DIR = pathlib.Path("/network/projects/")

    # PROJECT_DATA_DIR = pathlib.Path(os.getenv("BASE_DATA_PATH", ""))
    TREES_CO2_DIR = PROJECT_DATA_DIR / "trees-co2/quebec_trees/"
    ORTHO_DIR = TREES_CO2_DIR / "quebec_trees_dataset_2021-09-02/2021-09-02/"
    DSM_ORTHO_DIR = pathlib.Path(
        "/network/projects/trees-co2/quebec_trees/quebec_trees_dataset_2021-09-02/2021-09-02/"
    )
    DSM_ORTHOS_SAVE_DIR = TREES_CO2_DIR / "dsm_orthos_aligned"

    if not os.path.exists(DSM_ORTHOS_SAVE_DIR):
        os.makedirs(DSM_ORTHOS_SAVE_DIR)

    parcelles = ["zone1", "zone2", "zone3"]
    name_orthos = os.listdir(ORTHO_DIR)
    name_dsm_orthos = os.listdir(DSM_ORTHO_DIR)

    for p in parcelles:
        print("processing", p)
        folder = os.listdir(ORTHO_DIR / p)

        name_ortho = [i for i in folder if i.lower().find("-rgb-cog") != -1]
        name_dsm = [i for i in folder if i.lower().find("dsm") != -1]
        assert len(name_ortho) == 1
        assert len(name_dsm) == 1
        ORTHO_IMAGE = ORTHO_DIR / p / name_ortho[0]
        DSM_ORTHO_IMAGE = DSM_ORTHO_DIR / p / name_dsm[0]

        with rasterio.open(ORTHO_IMAGE) as ortho:
            # Save a lot of the information that is needed, so ortho image can be closed and
            # we can save space
            # ortho_data = ortho.read(1)  # Read the first band
            ortho_transform = ortho.transform
            ortho_crs = ortho.crs
            ortho_height = ortho.height
            ortho_width = ortho.width
            ortho_bounds = ortho.bounds
        print("height, width ortho", ortho_height, ortho_width)
        with rasterio.open(DSM_ORTHO_IMAGE) as dsm:
            dsm_crs = dsm.crs

            print("Check if CRS match")
            # Check if CRS match
            if dsm_crs != ortho_crs:
                print(ortho_crs, dsm_crs)
                # dsm= dsm.to_crs(ortho_crs)
                # dsm_crs = dsm.crs
                continue
                # raise ValueError("CRS does not match, reproject 'source' to match 'target'")

            print("Creating window to clip dsm ortho")
            window = from_bounds(*ortho_bounds, transform=dsm.transform)

            # Prepare to resample the source image
            kwargs = dsm.meta.copy()
            kwargs.update(
                {
                    "crs": dsm_crs,
                    "transform": ortho_transform,
                    "width": ortho_width,
                    "height": ortho_height,
                }
            )

            print("Resampling dsm ortho")

            with rasterio.open(
                DSM_ORTHOS_SAVE_DIR / name_dsm[0], "w", **kwargs
            ) as resampled:
                # Sample didn't include more than 1 band, but just in case...
                for i in range(1, dsm.count + 1):  # dsm.count + 1):
                    # Read each band from the source and resample it
                    resampled_band = dsm.read(
                        i,
                        window=window,
                        out_shape=(ortho_height, ortho_width),
                        resampling=Resampling.nearest,  # bilinear,
                        out_dtype="float32",
                    )
                    print(resampled_band.shape)
                    # print(resampled_band.height, resampled_band.width)
                    resampled.write(resampled_band, i)
            print("Done")
