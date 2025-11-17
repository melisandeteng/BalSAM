# SAM automatic baseline

This contains code to run inference with SAM in its automatic mode on tilerized orthomosaics and compute metrics.
We use Comet ML to log visualizations of of the predictions in the `compute_metrics.py`evaluation script. Set your `COMET_API_KEY` environment variable if you wish to log things into Comet.
We used the ViT-h backbone version of SAM and use the official implementation of [SAM](https://github.com/facebookresearch/segment-anything).

- First, run `infer_with_nms.py` to get SAM predictions using:
```
python infer_with_nms.py ROOT_PATH_TO_TILES OUTPUT_PATH --points_per_side 10 --fold "test"
``` 
The points-per-side parameter can be adjusted. In the paper, results were reported with pps=10 and pps=100.

- Then use `compute_metrics.py` to compute metrics on the SAM predictions.
```
python compute_metrics.py ROOT_PATH_TO_TILES OUTPUT_PATH --points_per_side 10 --fold "test" --log_comet --comet_project_name COMET_PROJECT
``` 
The arguments should match with the ones you used whem running `infer_with_nms.py`.