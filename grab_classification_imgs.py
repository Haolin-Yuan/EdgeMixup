# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

base_fname_dir = Path('NO_vs_EM_vs_HZ_vs_TC/')

train_df = pd.read_csv(str(base_fname_dir + 'NO_vs_EM_vs_HZ_vs_TC_train.csv'), header=None)
val_df = pd.read_csv(str(base_fname_dir + 'NO_vs_EM_vs_HZ_vs_TC_val.csv'), header=None)
test_df = pd.read_csv(str(base_fname_dir + 'NO_vs_EM_vs_HZ_vs_TC_test.csv'), header=None)

im_root_dir = Path('im_harvest_2019/images_July2019')
total_df = pd.concat([train_df, val_df, test_df])
total_df.columns = ['im_path', 'label']

d = {1: 'EM', 2: 'HZ', 3: 'TC'}
for lbl in d.keys():
    if not (Path.cwd() / d[lbl]).exists():
        (Path.cwd() / d[lbl]).mkdir()
    
    df = total_df[total_df['label'] == lbl]
    for i, path in enumerate(tqdm(df['im_path'])):
        im_path = im_root_dir / path
        img = cv2.imread(str(im_path))
        
        try:
            cv2.imwrite(str(Path.cwd() / d[lbl] / im_path.nacd ..me), img)
        except Exception as e:
            print(f"Saving {im_path} failed, skipping for now")
            continue
