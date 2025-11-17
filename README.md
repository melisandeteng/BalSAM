# Bringing SAM to new heights: leveraging elevation data for tree crown segmentation from drone imagery

This repository contains code to reproduce experiments and results of the paper ["Bringing SAM to new heights: leveraging elevation data for tree crown segmentation from drone imagery"](https://openreview.net/pdf?id=1vSLxdJNq8), by M. Teng, A. Ouaknine, E. Laliberté, Y. Bengio, D. Rolnick and H. Larochelle, accepted at the Thirty-ninth Annual Conference on Neural Information Processing Systems, NeurIPS 2025.

## Getting started

The code for this project is organized in two main parts. This is because the code for the RSPrompter and BalSAM models was built on top of the [original RSPrompter repository](https://github.com/KyanChen/RSPrompter) which uses the mmdet framework. The other baselines were coded using torchvision and transformers. You can find instructions for each model in their specific READMEs.

Dependencies for each part of the project are detailed in the `baseline_models` and `rsprompter_balsam` folders. 

### Data preparation
Code for these models with further instructions can be found in the folder `data_preparation`.  In particular, after reading `data_preparation/README.md`, refer to the `README.md` in `data_preparation/data_processing` folder. 

### SAM out-of-the-box, Faster R-CNN and Mask R-CNN-based models, Mask2Former model
Code for these models with further instructions can be found in the folder `baseline_models`.
It contains the code relevant for replicating the experiments using the following models:
- SAM automatic
- SAM + DSM prompts
- Mask R-CNN
- Mask R-CNN+SAM
- Faster R-CNN
- Mask2Former
Each subfolder model contains their corresponding relevant dataloading code. 

### RSPrompter and BalSAM models
Code for the RSPrompter and BalSAM models with further instructions can be found in the folder `rsprompter_balsam`. 

