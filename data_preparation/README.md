# Downloading the data and preparing it into tiles

This folder contains scripts to download and prepare the data from the [Quebec Plantations](https://www.frdr-dfdr.ca/repo/dataset/9f10a155-c89f-43ee-9864-da28ca436af6), [Quebec Trees](https://data.niaid.nih.gov/resources?id=zenodo_8148478) and [BCI](https://smithsonian.figshare.com/articles/dataset/Barro_Colorado_Island_50-ha_plot_crown_maps_manually_segmented_and_instance_segmented_/24784053?file=43628031)  datasets to reproduce the results from "Bringing SAM to new heights: leveraging elevation data for tree crown segmentation from drone imagery."

This data was prepared with the `geodataset v0.2.2` package.
Follow the instructions on the [geodataset repository](https://github.com/hugobaudchon/geodataset) to install the package.

You can use the same environment for preprocessing the data as the one used for the baselines that are not RSPrompter based. 

## Quebec Plantations dataset

- Download `Photogrammetry_Products` and `Vector_Data` folders of the [Quebec Plantations](https://www.frdr-dfdr.ca/repo/dataset/9f10a155-c89f-43ee-9864-da28ca436af6). You can remove data from Serpentin1 and Serpentin2 as they are not used in this study.
- Align the DSM orthomosaics with the RGB orthomosaics using `python align_dsm_orthos_plantations.py`
- The AOIs for preparing the tiles and assigning them to splits are in `../data/AOIs_plantations/`
- Use `geodataset_utils/tilerize_plantations.py` to tilerize the RGB orthomosaics  and `geodataset_utils/tilerize_dsm_plantations.py` to tilerize the DSM orthomosaics.
- Finally, to merge the annotations, run  `python merge_annotations.py`.

## Quebec Trees (SBL) dataset
- Download data from the 2023-09-12 date in the [Quebec Trees](https://data.niaid.nih.gov/resources?id=zenodo_8148478) dataset.
- Align the DSM orthomosaics with the RGB orthomosaics using `python align_dsm_orthos_sbl.py`
- The AOIs for preparing the tiles and assigning them to splits are in `../data/AOIs_sbl/`
- Use `geodataset_utils/tilerize_sbl.py` to tilerize the RGB orthomosaics  and `geodataset_utils/tilerize_dsm_sbl.py` to tilerize the DSM orthomosaics.
- Finally, merge the annotations with `python merge_annotations_sbl.py`.

## BCI dataset

Note that in this dataset, the orthomosaic tif has 4 channels, R,G,B and DSM. The DSM is encoded in uint8 and is 1m-vertical resolution.

- Download data from the [BCI](https://smithsonian.figshare.com/articles/dataset/Barro_Colorado_Island_50-ha_plot_crown_maps_manually_segmented_and_instance_segmented_/24784053?file=43628031) dataset:
  -  the imagery is in  `BCI_50ha_2022_09_29_crownmap_raw/tiles/BCI_50ha_2022_09_29_global.tif`
  - the annotations used for tilerizing were derived from the annotations in `BCI_50ha_2022_09_29_crownmap_improved` and were modified to use the categories in `../categories_subset_family.json`
- The AOIs for preparing the tiles and assigning them to splits are in `../data/AOIs_BCI/`
- Use `geodataset_utils/tilerize_bci.py` to tilerize the orthomosaic.
- To make this compatible with the format used for the other datasets, save the DSM tiles by opening the image tiles and saving just the 4th band as a DSM tile.
- Finally, merge the annotations with `merge_annotations_bci.py`.

