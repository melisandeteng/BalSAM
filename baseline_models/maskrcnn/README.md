# Mask R-CNN-based models.

This folder contains code for training a Mask R-CNN model using RGB or RGB+DSM inputs for tree crown instance segmentation on the BCI, Quebec Plantations and SBL datasets.
The `model.py`file also contains definitions offocal loss, hierarchical loss and weighted cross-entropy for running the ablation experiments on the SBL dataset. 

For each dataset, `plantations`, `sbl`, `bci`, we provide config files `config_{dataset_name}.yaml` and `config_{dataset_name}_dsm.yaml` for running models taking as input RGB or RGB+DSM in the `configs/` folder.

Our implementation is based on [torchvision object detection reference scripts](https://github.com/pytorch/vision/tree/main/references/detection). 

## Structure

## Training models

- Mask R-CNN with RGB inputs:  `train_maskrcnn.py`can be used to train Mask R-CNN models on all 3 datasets.

-  Mask R-CNN with RGB+DSM inputs: 
    - `train_maskrcnn_dsm.py` can be used to train Mask R-CNN+DSM models on the Quebec Plantations and SBL datasets
    - `train_maskrcnn_dsm_bci.py`can be used to train Mask R-CNN+DSM models on the BCI dataset

Example command line prompt: `python train_maskrcnn.py configs/config_plantations.yaml seed=0`

- Once you have trained models, you can refer to `inference.py` to run inference on the test sets of the datasets.
- `inference_sam_boxmask.py` provides an example script for using SAM with prompts that are outputs of a Mask R-CNN model for the Mask R-CNN+SAM baseline.

Example command line prompt: `python inference.py configs/config_plantations.yaml seed=0`


 

