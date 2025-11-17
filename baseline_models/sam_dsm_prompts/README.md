# SAM+DSM prompts baseline

In this folder is code to reproduce the SAM+DSM prompts baseline.
This is intended to be used on pre-processed orthomosaics (RGB and DSM) divided into tiles.
First, use ```get_local_maxima_{dataset_name}.py``` to get local maxima of the DSM for each dataset ("plantations", "sbl", "bci").
Make sure you change the paths to annotations (it should be the json annotation files in which the path to the DSM files are saved) and outputs. 

Then run ```infer_new_nms.py```to compute predictions of the SAM+DSM prompts model.
```
python infer_new_nms.py ROOT_PATH_TO_TILES HEIGHT_PROMPTS_PATH --fold "test" --output_path OUTPUT_PATH
```

Finally, run ```compute_new_nms_scores.py``` to evaluate the model: 
```
python compute_new_nms_scores.py ROOT_PATH_TO_TILES HEIGHT_PROMPTS_PATH --fold "test" --output_path OUTPUT_PATH
    parser.add_argument('--log_comet', action='store_true')
    parser.add_argument('--comet_project_name', default="sam_automatic_quebec_trees", help="comet project name")
```

