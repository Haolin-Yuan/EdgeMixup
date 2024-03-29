# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

#import config as cfg
from skin_segmentation.segmentation_cfg import *
from skin_segmentation.data_utils import Dataset, randomly_flip_image, generate_splits, load_splits
from skin_segmentation.model_utils import train_model, test_model, k_fold_split
from skin_segmentation.plot_utils import plot_images_w_masks, plot_history

def train_kfold(num_folds):
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()
    
    splits_list = []
    for img_root_dir, mask_json_dir in zip(IMG_ROOT_DIRS, cfg.MASK_JSON_DIRS):
        splits_list.append(k_fold_split(num_folds, img_root_dir, mask_json_dir, IMAGE_SIZE))

    splits = []
    for fold in range(num_folds):
        curr_split = {'train': None, 'val': None, 'test': None}
        for partition in curr_split.keys():
            for i in range(len(IMG_ROOT_DIRS)):
                if curr_split[partition]:
                    curr_split[partition] += splits_list[i][fold][partition]
                else:
                    curr_split[partition] = splits_list[i][fold][partition]
        splits.append(curr_split)

    scores_dict = {'split':[], 'iou_score':[], 'f1-score':[]}
    for i, split in enumerate(splits):
        print(f'=========RUNNING SPLIT {i + 1}=========')
        train_set, val_set, test_set = split['train'], split['val'], split['test']
        
        fold_output_dir = OUTPUT_DIR / f'split_{i + 1}'
        fold_model_path = fold_output_dir / 'best_model.h5'
        if not fold_output_dir.exists():
            fold_output_dir.mkdir()

        hist = train_model(train_set, val_set, BATCH_SIZE, BACKBONE, EPOCHS, LR, fold_model_path)
        plot_history(hist, fold_output_dir)

        scores, names = test_model(test_set, fold_model_path, fold_output_dir, BATCH_SIZE)
        #plot_images_w_masks(test_set, fold_model_path, fold_output_dir, num_plots='all', save_plots=True)

        print(f'=========SCORES FOR SPLIT {i + 1}=========')
        scores_dict['split'].append(i + 1)
        for metric, val in zip(names[1:], scores[1:]):
            print(f"Mean {metric}: {val}\n")
            if scores_dict[metric]:
                scores_dict[metric].append(val)
            else: 
                scores_dict[metric] = [val]

    scores_dict['Split'].append('Average')
    for metric, val in zip(list(scores_dict.keys())[1:], list(scores_dict.values())[1:]):
        scores_dict[metric].append(np.mean(val))

    scores_df = pd.DataFrame(scores_dict)
    scores_df.to_csv(str(OUTPUT_DIR / 'scores.csv'), columns=scores_df.columns, index=False)




def train_normal(mode):
    train_set, val_set, _ = load_splits(edgemixup=mode)
    hist = train_model(train_set, val_set, BATCH_SIZE, BACKBONE, EPOCHS, LR, MODEL_PATH)
    print("printing model training hist:", hist)
    # plot_history(hist, OUTPUT_DIR)

def test(mode):
    _, _, test_set = load_splits(edgemixup=mode)
    scores, names = test_model(test_set, OUTPUT_DIR, BATCH_SIZE)
    
    scores_dict = {key: [] for key in names[1:]}
    print("-"*10)

    for metric, val in zip(list(scores_dict.keys()), scores[1:]):
        print(f"{metric}: {val}\n")
        scores_dict[metric].append(val)

    scores_df = pd.DataFrame(scores_dict)
    scores_df.to_csv(str(OUTPUT_DIR / 'scores.csv'), columns=scores_df.columns, index=False)

    # _, _, test_set = load_splits(split=True, return_raw_imgs=True)
    # plot_images_w_masks(test_set, MODEL_PATH, OUTPUT_DIR, num_plots='all', save_plots=True)
    return scores_dict["jaccard_score"]

def train_iterative():
    print("========Iteratively training start========")
    pre_J_score = 0
    curr_J_score = 0
    round_count = 1
    while curr_J_score>= pre_J_score:
        print(f"Iterative Training round {round_count}")
        train_normal(mode=True)
        new_J = test(mode=True)[0]
        if round_count == 1:
            pre_J_score = new_J
            curr_J_score = new_J
        else:
            pre_J_score = curr_J_score
            curr_J_score = new_J
        round_count += 1
        print(f"Last iteration Jaccard: {pre_J_score}, Current iteration Jaccard:{curr_J_score}")
    print(f"Iteration training stopped, {round_count} iterations have been trained, last iteration jaccard:{pre_J_score}")


def plot_all():
    all_set = load_splits(split=False, return_raw_imgs=True)
    plot_images_w_masks(all_set, MODEL_PATH, OUTPUT_DIR, num_plots='all', save_plots=True, save_subdir="full_dataset_plots")

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-splits", action="store_true", help="Generate new train/validation/test splits.")
    parser.add_argument("--train", action="store_true", help="Train and plot training/val results.")
    parser.add_argument("--kfold-train", action="store_true", help="Train using KFold.")
    parser.add_argument("--test", action="store_true", help="Test only.")
    parser.add_argument("--plot-all", action="store_true", help="Generate masks for the entire (train+val+test) dataset.")
    parser.add_argument("--iterative", action="store_true", help="iteratively train the segmentation model")
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.regenerate_splits:
        generate_splits()

    if args.train:
        # by default EdgeMixup is not used
        train_normal(mode=False)
    
    if args.kfold_train:
        train_kfold(num_folds=5)
            
    if args.test:
        test(args.edgemixup)
    
    if args.plot_all:
        plot_all()

    if args.iterative:
        train_iterative()


if __name__=='__main__':
    main()

