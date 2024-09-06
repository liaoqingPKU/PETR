import argparse
import copy
import mmcv
import os
import sys
import time
import torch
import warnings
import numpy as np
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.onnx import register_extra_symbolics
import pickle as pkl
from tqdm import tqdm

import onnx
import onnxsim
from onnxsim inport simplify

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

def load_model(config_path, save_folder, ckpts_path=False):
    print('[LOADING CONFIG...PATH:{}]'.format(config_path))
    
