# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

from config import *

# Classification config options
ROOT_CLF_IMG_DIR = Path()
CLF_ASSETS = Path('../disease_clf_assets')
CLF_RAW_SPLITS_DIR = CLF_ASSETS / 'raw_splits'
CLF_SPLITS_DIR = CLF_ASSETS / 'processed_splits'
CLF_OUTPUT_DIR = Path('disease_classification/classification_runs')
CLF_MASK_DIR = CLF_OUTPUT_DIR / 'masks'

CLF_BATCH_SIZE = 32
CLF_LR = 1.e-3
CLF_NUM_EPOCHS = 100
CLF_LR_STEP_FACTOR = 0.1
CLF_LR_SCHEDULER_PATIENCE = 10
CLF_ADV_LOSS_IMPORTANCE = 1.0

CLF_Label_Translate = {
    0: 'NO',
    1: 'EM',
    2: 'HZ',
    3: 'TC'
}

NUM_DISEASE_CLASSES = 4
NUM_PROTECTED_CLASSES = 2