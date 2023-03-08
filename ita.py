# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import time
import math
import argparse
from pathlib import Path

import cv2
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from skin_segmentation.model_utils import predict_mask_generator, calculate_ITA_test_metrics
from skin_segmentation.data_utils import load_splits, ITADataLoader, calc_batch_ITAs
from ita_data.define_ita_bins import get_ita_data
#from config import *
import config as cfg
import disease_classification.clf_config as clf_cfg

import pdb


def calculate_ita(split=True):
    if split:
        _, _, dataset = load_splits(split=True, return_raw_imgs=True)
    else:
        dataset = load_splits(split=False, return_raw_imgs=True)
    #img_paths = [str(path) for path in IMG_ROOT_DIRS.iterdir() if path.is_file()]  # grabs a list of the image paths
    df = pd.DataFrame(columns=['image', 'gt_category', 'pred_category', 'gt_ita', 'pred_ita'])  # creates the dataframe
    ita_bins_data = get_ita_data()  # loads in the data for each ITA category
    # Predicting all masks at once is too memory intensive, so we do it in batches
    preds = predict_mask_generator(dataset, batch_size=cfg.SKIN_SEG_BATCH_SIZE, make_masks_usable=False)

    loader = ITADataLoader(dataset, preds, batch_size=cfg.SKIN_SEG_BATCH_SIZE)

    s = time.time()
    #pbar = trange(0, len(all_set), cfg.SKIN_SEG_BATCH_SIZE, desc="Categorizing images")
    for content in tqdm(loader, desc="Categorizing images"):
        '''
        _, _, img_batch, img_idxs = content
        preds_batch = preds[i : i + cfg.SKIN_SEG_BATCH_SIZE]
        img_paths = [all_set.im_paths[j] for j in img_idxs.tolist()]
        '''
        batch_raw_imgs, batch_img_paths, batch_gt_masks, pred_masks = content
        
        #img_loader, preds = predict_mask_generator(img_loader, batch_size=cfg.SKIN_SEG_BATCH_SIZE, make_masks_usable=False)
        #pred_masks = bounded_decision_function(preds_batch, T=.65) # TODO Make this a config option or something
        
        pred_ita_vals = calc_batch_ITAs(batch_raw_imgs, pred_masks)
        batch_gt_masks = np.expand_dims(np.argmax(batch_gt_masks, axis=-1), axis=-1)
        gt_ita_vals = calc_batch_ITAs(batch_raw_imgs, batch_gt_masks)

        # Saves the skin tone category for each image to the dataframe (categories defined in
        # skin_tone_categorization_scheme.txt, which is pulled from Table 1 in the original paper).
        for img_path, pred_ita, gt_ita in zip(batch_img_paths, pred_ita_vals, gt_ita_vals):
            pred_tone = 'undef'
            gt_tone = 'undef'
            for k in ita_bins_data.keys():
                if ita_bins_data[k]['check_in_bin'](pred_ita):  # checks which bin the current ITA value falls under
                    pred_tone = k
                if ita_bins_data[k]['check_in_bin'](gt_ita):  # checks which bin the current ITA value falls under
                    gt_tone = k

            df = df.append({'image':str(Path(img_path).name), 'gt_category':gt_tone, 'pred_category':pred_tone, 'gt_ita':gt_ita, 'pred_ita':pred_ita}, ignore_index=True) # Apparently df.append() doesn't append in place...y tho

    csv_name = str(cfg.SKIN_TONE_CSV).replace('skin_tones', 'test_skin_tones') if split else str(cfg.SKIN_TONE_CSV)
    df.to_csv(csv_name, columns=df.columns, index=False) # save each iteration in case errors occur later

    print(f"{len(dataset)} images categorized in {time.time() - s} seconds")


def generate_ita_distribution(dataset_csvs=None, show_plot=False):
    # dataset_csvs = []
    # for csv in (TRAIN_CSV, VAL_CSV, TEST_CSV):
    #     if csv is not None:
    #         dataset_csvs.append(csv)

    csv_name = str(cfg.SKIN_TONE_CSV).replace('skin_tones', 'test_skin_tones')
    skin_tone_df = pd.read_csv(csv_name)
    save_path = cfg.OUTPUT_DIR / 'ita_distribution.png'
    bar_labels = list(get_ita_data().keys()) #alternative bar_labels = np.unique(skin_tone_df['gt_category'])

    if dataset_csvs is not None:
        reference_dfs = {}
        for csv_path in dataset_csvs:
            reference_dfs[csv_path.stem] = pd.read_csv(str(csv_path))

        counts_by_df = np.zeros(shape=(len(reference_dfs.keys()), len(bar_labels)), dtype=int)
        for i, label in enumerate(bar_labels):
            ims_w_curr_label = list(skin_tone_df[skin_tone_df['gt_category'] == label]['image'])
            for j, ref_df in enumerate(reference_dfs.values()):
                ims_from_df = ref_df[ref_df['image'].isin(ims_w_curr_label)]
                counts_by_df[j][i] += len(ims_from_df)


        fig, ax = plt.subplots(1, 1)

        plots = []
        for i, counts in enumerate(counts_by_df):
            if i == 0:
                plots.append(ax.bar(bar_labels, counts))
            else:
                bottom = np.sum(counts_by_df[np.arange(len(counts_by_df)) < i], axis=0)
                plots.append(ax.bar(bar_labels, counts, bottom=bottom))

        ax.set_title('Distribution of ITA Categories in Dataset')
        ax.set_xlabel("ITA Categories")
        ax.legend([plot[0] for plot in plots], list(reference_dfs.keys()))

        total_counts = np.sum(counts_by_df, axis=0)
        for i, v in enumerate(total_counts):
            v_offset = max(total_counts) // 100
            ax.text(i, v + v_offset, str(v), color='black', fontweight='bold', ha='center')

    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)

        gt_vals, raw_gt_counts = np.unique(skin_tone_df['gt_category'], return_counts=True)
        #gt_counts = [gt_counts[list(gt_vals).index(label)] if label in gt_vals else 0 for label in bar_labels]
        pred_vals, raw_pred_counts = np.unique(skin_tone_df['pred_category'], return_counts=True)
        
        gt_counts, pred_counts = [], []
        for label in bar_labels:
            if label in gt_vals:
                gt_counts.append(raw_gt_counts[list(gt_vals).index(label)])
            else:
                gt_counts.append(0)
            if label in pred_vals:
                pred_counts.append(raw_pred_counts[list(pred_vals).index(label)])
            else:
                pred_counts.append(0)
        
        width = 0.5
        X_idxs = np.arange(len(bar_labels))
        ax.bar(X_idxs - width / 2., gt_counts, width=width, color='g', align='center', label="GT")
        ax.bar(X_idxs + width / 2., pred_counts, width=width, color='orange', align='center', label="pred")
        ax.set_title('Distribution of ITA Categories in Test Dataset')
        ax.set_xlabel("ITA Categories")
        for i in range(len(gt_counts)):
            gt_v = gt_counts[i]
            pred_v = pred_counts[i]
            gt_v_offset = max(gt_counts) // 100
            ax.text(i - width / 2., gt_v + gt_v_offset, str(gt_v), color='g', fontweight='bold', ha='center')
            pred_v_offset = max(pred_counts) // 100
            ax.text(i + width / 2., pred_v + pred_v_offset, str(pred_v), color='orange', fontweight='bold', ha='center')
        ax.legend()
        plt.xticks(X_idxs, bar_labels)
        #ax.autoscale(tight=True)

    fig.savefig(str(save_path))

    if show_plot:
        plt.show()
    plt.close()

def calculate_bias_metrics(split=True):
    assert cfg.SKIN_TONE_CSV.is_file, "Calculate ITA before calculating the bias metrics."
    skin_tone_df = pd.read_csv(str(cfg.SKIN_TONE_CSV))

    if split:
        _, _, dataset = load_splits(split=split, return_raw_imgs=True)
    else:
        dataset = load_splits(split=split, return_raw_imgs=True)
    
    all_metrics = calculate_ITA_test_metrics(dataset, skin_tone_df)


def generate_clf_ita_distribution(dataset_csvs=None, show_plot=False):
    split = "test"
    csv_name = clf_cfg.CLF_SPLITS_DIR / f"{split}_processed.csv"
    skin_tone_df = pd.read_csv(str(csv_name))
    save_path = clf_cfg.CLF_OUTPUT_DIR / f'{split}_ita_distribution.png'
    bar_labels = list(get_ita_data().keys()) #alternative bar_labels = np.unique(skin_tone_df['gt_category'])

    fig, ax = plt.subplots(nrows=1, ncols=1)

    pred_vals, raw_pred_counts = np.unique(skin_tone_df['pred_skintone_category'], return_counts=True)
    
    gt_counts, pred_counts = [], []
    for label in bar_labels:
        if label in pred_vals:
            pred_counts.append(raw_pred_counts[list(pred_vals).index(label)])
        else:
            pred_counts.append(0)
    
    width = 0.5
    X_idxs = np.arange(len(bar_labels))
    ax.bar(X_idxs + width / 2., pred_counts, width=width, color='orange', align='center', label="pred")
    #ax.set_title('Distribution of ITA Categories in the Classification Test Dataset')
    ax.set_xlabel("ITA Categories")
    for i in range(len(gt_counts)):
        pred_v = pred_counts[i]
        pred_v_offset = max(pred_counts) // 100
        ax.text(i + width / 2., pred_v + pred_v_offset, str(pred_v), color='orange', fontweight='bold', ha='center')
    plt.xticks(X_idxs, bar_labels)

    fig.savefig(str(save_path))
    if show_plot:
        plt.show()
    plt.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calculate-ita", action="store_true", help="Calculate the ITA for the ground truth masks and the predicted masks.")
    parser.add_argument("--plot", action="store_true", help="Generate ITA distribution plot.")
    parser.add_argument("--bias-metrics", action="store_true", help="Calculate the bias metrics using the ITA.")
    parser.add_argument("--plot-classification", action="store_true", help="Generate classification dataset ITA distribution plot.")
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.calculate_ita:
        calculate_ita(split=True)

    if args.plot:
        generate_ita_distribution()
    
    if args.bias_metrics:
        calculate_bias_metrics()
    
    if args.plot_classification:
        generate_clf_ita_distribution()


if __name__ == '__main__':
    main()
