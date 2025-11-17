# Faster R-CNN + SAM baseline

This folder contains code to train a Faster R-CNN on img or img+DSM inputs and get predictions of SAM when it is given the outputs of a trained Faster R-CNN as prompts. 

Please refer to the example config files in the `configs/` folder. 

Faster R-CNN Training: 
- RGB inputs: `python train_fasterrcnn.py CONFIG_FILE_PATH  OVERRIDING_ARGUMENTS(e.g. seed=0)`
- RGB+DSM inputs: `python train_fasterrcnn_dsm.py CONFIG_FILE_PATH OVERRIDING_ARGUMENTS`
Inference: inference with Faster R-CNN + passes box predictions as prompts to SAM" 
`python inference.py CONFIG_FILE_PATH`