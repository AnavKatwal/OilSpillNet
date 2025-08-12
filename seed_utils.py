import os, random, numpy as np, tensorflow as tf

def set_global_seed(seed=2023):
    # set seeds for os environments
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # set seeds for built-in libraries
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)