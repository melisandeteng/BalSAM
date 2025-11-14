# Mask2former baseline

We use the Mask2Former implementation in the `transformers` package. 

To train the models: 
- Make sure to modify YOUR_COMET_API_KEY and YOUR_COMET_WORKSPACE at the top of the the training and inference .py files, and to change the dataset and output saving paths. 
- We provide training script for Mask2Former with or without DSM input, using Swin-base or Swin-L backbones. 
    - `mask2former-base.py`: Mask2Former with RGB input, Swin-base backbone
    - `mask2former-base_dsm.py`: Mask2Former with RGB+DSM input, Swin-base backbone
    - `mask2former-large.py`: Mask2Former with RGB input, Swin-L backbone
    - `mask2former-large_dsm.py`: Mask2Former with RGB+DSM input, Swin-L backbone

- Example command line: `python mask2former-base.py`

Mask2Former baselines were only run on the Plantations dataset. 

We also provide example evaluation scripts in `metrics_mask2former-base.py` and `metrics_mask2former-dsm_base.py`.

