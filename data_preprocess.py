import difflib
import javalang.tokenizer as java_tokenizer
import pandas as pd
import numpy as np
from code2vec_utils import download_code2vec
from javalang.tree import Literal, MemberReference, ConstructorDeclaration, IfStatement, WhileStatement, ForStatement
from javalang.parser import Parser as JavaParser
from tqdm import tqdm
from pathlib import Path
import sys
import os
sys.path.append(str(Path(".", "code2vec").resolve()))
from code2vec import code2vec, config, extractor, common

PARSE_RESULTS = dict()
VECTOR_EMBEDDINGS = dict()

# code2vec config options
SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'

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
            shared_results[f"{key}_diff"] = np.abs(s1_parse_results[key] - s2_parse_results[key])
        elif s1_parse_results[key] == s2_parse_results[key]:
            shared_results[f"same_{key}"] = 1
        else:
            shared_results[f"same_{key}"] = 0
    return shared_results

def get_vector_embedding(example, func_num, path_extractor, model):
    if example[f"id{func_num}"] not in VECTOR_EMBEDDINGS:
        with open("code2vec/Input.java", "w", encoding='utf-8') as f:
            f.write(example[f"func{func_num}"])
        predict_lines, hash_to_string_dict = path_extractor.extract_paths("code2vec/Input.java")
        raw_prediction_results = model.predict(predict_lines)
        VECTOR_EMBEDDINGS[example[f"id{func_num}"]] = raw_prediction_results[0].code_vector
    return VECTOR_EMBEDDINGS[example[f"id{func_num}"]]

def merge_dicts(*dict_args):
    result = dict()
    for dictionary in dict_args:
        result.update(dictionary)
    return result 

def big_clone_bench_preprocess(bcb, csv_filename=None, enable_code2vec=False):
    '''
    Generates DataFrame of features from an input in the format of the BigCloneBench dataset https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench
    '''
    df = pd.DataFrame()
    if enable_code2vec:
        if not os.path.exists('./code2vec/models/java14_model'):
            download_code2vec()
        c2v_config = config.Config(set_defaults=True)
        c2v_config.MODEL_LOAD_PATH = './code2vec/models/java14_model/saved_model_iter8.release'
        c2v_config.EXPORT_CODE_VECTORS = True
        c2v_config.DL_FRAMEWORK = 'tensorflow'

        model = code2vec.load_model_dynamically(c2v_config)

        path_extractor = extractor.Extractor(c2v_config,
                                    jar_path=JAR_PATH,
                                    max_path_length=MAX_PATH_LENGTH,
                                    max_path_width=MAX_PATH_WIDTH)

    print(f"Preprocessing {len(bcb)} examples...")
    for i in tqdm(range(len(bcb))):
        example = bcb[i]
        example_dict = line_diff(example["func1"], example["func2"])
        func1_parse_results = get_parse_results(example, 1)
        func2_parse_results = get_parse_results(example, 2)
        if enable_code2vec:
            func1_vector_embedding = get_vector_embedding(example, 1, path_extractor, model)
            func2_vector_embedding = get_vector_embedding(example, 2, path_extractor, model)
            embedding_distance = np.linalg.norm(func1_vector_embedding - func2_vector_embedding)
            example_dict["embedding_dist"] = embedding_distance
        shared_results = combine_parse_results(func1_parse_results, func2_parse_results)
        example_dict = merge_dicts(example_dict, shared_results)
        example_dict["target"] = int(example["label"])
        df = pd.concat([df, pd.DataFrame([example_dict])], ignore_index=True)
    if csv_filename != None:
        df.to_csv(csv_filename)
    return df
             
