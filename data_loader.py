from datasets import load_dataset

def load_big_clone_bench(split=None):
    bcb = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
    if split == None:
        return bcb
    else:
        return bcb[split]