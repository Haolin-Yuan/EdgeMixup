# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

from pathlib import Path
from config import *
import numpy as np

RANDOM_STATE = 2020
#
IMG_RETRIVAL_DATESTAMP = "01_06_2021"
ROOT_DATA_DIR = Path(f"/home/haolin/Projects/lyme/lyme_segmentation_data/{IMG_RETRIVAL_DATESTAMP}")
OUTPUT_DIR = Path("/home/haolin/Projects/lyme/code/skin_segmentation/segmentation_runs/test")
#
IMG_ROOT_DIRS = [d / 'img' for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
MASK_JSON_DIRS =  [d / 'ann' for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
MODEL_PATH = OUTPUT_DIR / 'best_model.h5'
DATA_ASSETS_DIR = Path(".") / str(ROOT_DATA_DIR).split("/")[-1]
SPLIT_PERCENTAGES = {'train': .65, 'val': .15, 'test': .20}
BACKBONE = 'resnet50'
RETURN_SUMMARY = True
BATCH_SIZE = 8 # 32
LR = 0.0001
STOPPING_PATIENCE = 8
EPOCHS = 5
IMAGE_SIZE = (256, 256)
SEG_LOSS_BETA = 1.0
METRIC_BETA = 1.0

# Loss Terms
DICE_LOSS_WEIGHTS = np.array([0.5, 2, 1]) # BG, LESION, SKIN
FOCAL_LOSS_WEIGHTS = np.array([0.15, 0.5, 0.35]) # BG, LESION, SKIN, must sum to 1.0

# EdgeMixup
Iter_root = "/home/haolin/Projects/lyme_demo/iter_seg_train_data/"
