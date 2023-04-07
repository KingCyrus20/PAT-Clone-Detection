import difflib
import javalang.tokenizer as java_tokenizer
import pandas as pd
from javalang.tree import Literal, MemberReference, ConstructorDeclaration, IfStatement, WhileStatement, ForStatement
from javalang.parser import Parser as JavaParser
from tqdm import tqdm

PARSE_RESULTS = dict()

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
   
def parse_java(java_string):
    parse_results = dict()
    tokens = java_tokenizer.tokenize(java_string)
    parser = JavaParser(tokens)
    ast = parser.parse_member_declaration()
    nodes = [node for _, node in ast]
    parse_results["num_literals"] = 0
    parse_results["num_if_statements"] = 0
    parse_results["num_while_statements"] = 0
    parse_results["num_for_statements"] = 0
    unique_identifiers = set()

    for node in nodes:
        if isinstance(node, Literal):
            parse_results["num_literals"] += 1
        elif isinstance(node, IfStatement):
            parse_results["num_if_statements"] += 1
        elif isinstance(node, WhileStatement):
            parse_results["num_while_statements"] += 1
        elif isinstance(node, ForStatement):
            parse_results["num_for_statements"] += 1
        elif isinstance(node, MemberReference):
            unique_identifiers.add(node.member)

    if isinstance(ast, ConstructorDeclaration):
        parse_results["is_constructor"] = "True"
    else:
        parse_results["is_constructor"] = "False"

    if "parameters" in ast.attrs:
        parse_results["num_parameters"] = len(ast.parameters)
    else:
        parse_results["num_parameters"] = 0

    if "return_type" in ast.attrs:
        parse_results["return_type"] = ast.return_type
    else:
        parse_results["return_type"] = None

    if "throws" in ast.attrs and ast.throws != None:
        parse_results["num_throws"] = len(ast.throws)
    else:
        parse_results["num_throws"] = 0
        
    parse_results["num_identifiers"] = len(unique_identifiers)

    return parse_results

def get_parse_results(example, func_num):
    if example[f"id{func_num}"] not in PARSE_RESULTS:
        PARSE_RESULTS[example[f"id{func_num}"]] = parse_java(example[f"func{func_num}"])
    return PARSE_RESULTS[example[f"id{func_num}"]]

def combine_parse_results(s1_parse_results, s2_parse_results):
    shared_results = dict()
    for key in s1_parse_results:
        if isinstance(s1_parse_results[key], int):
            shared_results[f"{key}_s1"] = s1_parse_results[key]
            shared_results[f"{key}_s2"] = s2_parse_results[key]
        elif s1_parse_results[key] == s2_parse_results[key]:
            shared_results[f"same_{key}"] = 1
        else:
            shared_results[f"same_{key}"] = 0
    return shared_results

def merge_dicts(*dict_args):
    result = dict()
    for dictionary in dict_args:
        result.update(dictionary)
    return result 

def big_clone_bench_preprocess(bcb, csv_filename=None):
    '''
    Generates DataFrame of features from an input in the format of the BigCloneBench dataset https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench
    '''
    df = pd.DataFrame()
    print(f"Preprocessing {len(bcb)} examples...")
    for i in tqdm(range(len(bcb))):
        example = bcb[i]
        example_dict = line_diff(example["func1"], example["func2"])
        func1_parse_results = get_parse_results(example, 1)
        func2_parse_results = get_parse_results(example, 2)
        shared_results = combine_parse_results(func1_parse_results, func2_parse_results)
        example_dict = merge_dicts(example_dict, shared_results)
        example_dict["target"] = int(example["label"])
        df = pd.concat([df, pd.DataFrame([example_dict])], ignore_index=True)
    if csv_filename != None:
        df.to_csv(csv_filename)
    return df
             
