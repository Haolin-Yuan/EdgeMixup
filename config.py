# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

from enum import Enum
from pathlib import Path
import numpy as np

# Config options used in all parts of the pipeline
IMAGE_SIZE = (256, 256)
IMG_RETRIVAL_DATESTAMP = "03_03_2023" #"01_06_2021" #"11_25_2020"# TODO: change this back to this "01_06_2021" #"12_16_2020" #"11_25_2020"
ROOT_DATA_DIR = Path(f"lyme_segmentation_data/{IMG_RETRIVAL_DATESTAMP}")

ITA_JSON = Path.cwd() / "ita_data" / "ita_cutoffs.json"
OUTPUT_DIR = Path.cwd() / "skin_segmentation" / "segmentation_runs" / f"test_output_{IMG_RETRIVAL_DATESTAMP}_TC_HZ_EM"
SKIN_TONE_CSV = OUTPUT_DIR / 'skin_tones.csv'

# Config options for skin segmentation and ITA calculation
SKIN_SEG_BATCH_SIZE = 8
BOUNDED_DECISION_THRESHOLD = .65

class Label(Enum):
    BG = 0
    LESION = 1
    SKIN = 2

# ################
alpha = 0.7
AUX_DATA_DIR = Path("lyme_segmentation_data/AUX")
AUX_MODEL_DIR = None
NUMB_CLASS = 0


