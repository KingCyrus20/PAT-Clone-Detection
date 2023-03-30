import difflib
import javalang
import pandas as pd
from tqdm import tqdm

def line_diff(s1: str, s2: str, match_speed = "real_quick"):
    differ = difflib.Differ()
    diff = differ.compare(s1.splitlines(True), s2.splitlines(True))
    result = {"unique_s1_lines": 0, "unique_s2_lines": 0, "shared_lines": 0}
    for line in diff:
        if line.startswith("- "):
            result["unique_s1_lines"] += 1
        elif line.startswith("+ "):
            result["unique_s2_lines"] += 1
        elif line.startswith("  "):
            result["shared_lines"] += 1
    seq_match = difflib.SequenceMatcher(None, s1, s2)
    if match_speed == "slow":
        result["similarity_ratio"] = seq_match.ratio()
    elif match_speed == "quick":
        result["similarity_ratio"] = seq_match.quick_ratio()
    elif match_speed == "real_quick":
        result["similarity_ratio"] = seq_match.real_quick_ratio()
    return result

# TODO: generate features from Java ast    
# def parse_java(java_string):
#     parse_results = dict()
#     tree = javalang.parse.parse(java_string)
#     return parse_results

def big_clone_bench_preprocess(bcb, csv_filename=None):
    '''
    Generates DataFrame of features from an input in the format of the BigCloneBench dataset https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench
    '''
    df = pd.DataFrame()
    # parse_results = dict()
    print(f"Preprocessing {len(bcb)} examples...")
    for i in tqdm(range(len(bcb))):
        example = bcb[i]
        example_dict = line_diff(example["func1"], example["func2"])
        example_dict["target"] = int(example["label"])
        # if example["id1"] not in parse_results:
        #     parse_results[example["id1"]] = parse_java(example["func1"])
        # if example["id2"] not in parse_results:
        #     parse_results[example["id2"]] = parse_java(example["func2"])
        df = pd.concat([df, pd.DataFrame([example_dict])], ignore_index=True)
    if csv_filename != None:
        df.to_csv(csv_filename)
    return df
             
