# PAT-Clone-Detection
ML-based code clone detection program for COSC 6386

## Usage
First, install requirements
```
pip install requirements.txt
```
Then, run
```
python clone_detection.py
```
This will run the model selected by the `--classifier` option (defaults to `random_forest`) on the BigCloneBench validation and test datasets.
For generating predictions on a custom dataset, pass the path to a csv file in the format of the [BigCloneBench dataset](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench) to the `--custom_data` option.
