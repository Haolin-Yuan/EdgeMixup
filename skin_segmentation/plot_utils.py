# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import os
import sys
from pathlib import Path
sys.path.append("..")

import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from segmentation_models.metrics import IOUScore
from sklearn.metrics import jaccard_score, fbeta_score
from tqdm import trange
import keras
import tensorflow as tf
from keras import backend as K

# from lyme_segmentation.config import Label
import skin_segmentation.segmentation_cfg as cfg
from skin_segmentation.model_utils import predict_mask, split_mask, make_mask_usable
from skin_segmentation.data_utils import bounded_decision_function
from ita import calculate_ita

import pdb

def plot_images_w_masks(dataset, model_path, output_dir, num_plots='all', T=None, save_plots=False, show_plots=False, save_subdir="predictions"):
    calculate_ita(split=False)
    ita_df = pd.read_csv(str(cfg.SKIN_TONE_CSV))
    
    if num_plots == 'all':
        num = len(dataset)
    else:
        num = num_plots

    all_fnames = {}
    save_dir = output_dir / save_subdir
    if save_plots:
        if not save_dir.exists():
            save_dir.mkdir()

    for i in trange(num):
        img, gt_mask, raw_img = dataset[i][0], np.argmax(dataset[i][1], axis=-1), dataset[i][2]
        
        _, model_pred = predict_mask(img, model_path=model_path)

        if T:
            pr_mask = bounded_decision_function(model_pred, T=T)
        else:
            pr_mask = make_mask_usable(model_pred)
        
        gt_flat = keras.utils.to_categorical(gt_mask)
        pr_flat = keras.utils.to_categorical(pr_mask)
        
        gt_flat = gt_flat.reshape(-1, len(cfg.Label))
        pr_flat = pr_flat.reshape(-1, len(cfg.Label))
        iou = jaccard_score(
            y_true=gt_flat,
            y_pred=pr_flat,
            average='macro')
        '''
        iou = IOUScore(threshold=0.5)(, )
        with tf.Session() as sess:
            K.set_session(sess)
            iou = sess.run(iou)
        keras.backend.clear_session()
        '''

        # skin_mask, lesion_mask = split_mask(pr_mask)
        # skin_plots = {"Image": img, "Skin Mask": np.squeeze(skin_mask), "Combined": img * skin_mask}
        # lesion_plots = {"Image": img, "Lesion Mask": np.squeeze(lesion_mask), "Combined": img * lesion_mask}

        fig, ((orig_plot, gt_mask_plot), (pr_overlay, pr_mask_plot)) = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
        # fig.suptitle(f"IOU Score: {iou}\nBackground is black, Skin is pink, Lesion is yellow")
        img_pth = Path(dataset.im_paths[i])
        img_name = img_pth.name
        img_subdir = str(img_pth.parents[1]).split("/")[-1]

        ita_idx = ita_df.iloc[ita_df[ita_df['image'] == str(img_name)].index]
        gt_ita = ita_idx['gt_ita'].values[0]
        pred_ita = ita_idx['pred_ita'].values[0]

        #fig.suptitle(f"IOU Score: {iou:3f} Ref Image: {img_subdir}/{img_name}\nBackground is purple, Skin is yellow, Lesion is blue")
        fig.suptitle(f"IOU Score: {iou:3f}\nBackground is purple, Skin is yellow, Lesion is blue")

        orig_plot.imshow(raw_img)
        orig_plot.set_xticks([])
        orig_plot.set_yticks([])
        orig_plot.set(title="Image")

        gt_mask = gt_mask * 127
        gt_mask_plot.imshow(gt_mask)
        gt_mask_plot.set_xticks([])
        gt_mask_plot.set_yticks([])
        gt_mask_plot.set(title=f"GT Mask (ITA: {gt_ita:3f})")

        # cmap = colors.ListedColormap(['black', 'yellow', 'purple'])
        # bounds = np.linspace(0, 3, 4)
        # norm = colors.BoundaryNorm(bounds, cmap.N)

        pr_overlay.imshow(raw_img)
        pr_mask = pr_mask * 127 # Map range 0-2 to 0-254
        pr_overlay.imshow(np.squeeze(pr_mask), alpha=.4)
        pr_overlay.set_xticks([])
        pr_overlay.set_yticks([])
        pr_overlay.set(title="Pred Mask and Image Overlay")

        pr_mask_plot.imshow(pr_mask)
        pr_mask_plot.set_xticks([])
        pr_mask_plot.set_yticks([])
        pr_mask_plot.set(title=f"Pred Mask (ITA: {pred_ita:3f})")

        if save_plots:
            fname = save_dir / f'output{i}.png'
            fig.savefig(str(fname))
            all_fnames[iou] = fname

        if show_plots:
            plt.show()
        
        plt.cla()
    
    # Move the worst performers to their own directory. Only do this if working on all of the images
    if save_plots and save_subdir != "predictions":
        # Move the worst performers
        sorted_fnames = [v for k, v in sorted(all_fnames.items(), key=lambda item: item[0])]
        worst_performers_dir = save_dir / "worst"
        if not worst_performers_dir.exists():
            worst_performers_dir.mkdir()
        
        for fname in sorted_fnames[:40]:
            new_fname = worst_performers_dir / fname.name
            os.rename(str(fname), str(new_fname))

def plot_history(hist, ouput_dir):
    beta = int(cfg.METRIC_BETA)
    hist_metrics = hist.history

    metrics = [metric for metric in hist_metrics.keys()]
    print(metrics)

    epochs = np.arange(len(hist_metrics["loss"]))
    # Jaccard/IoU
    fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, hist_metrics["jaccard_score"], label="SKLearn_jaccard_score")
    ax.plot(epochs, hist_metrics["iou_score"], label="SegLib_iou_score")
    ax.set(xlabel='Epochs', ylabel='IoU Score', title="Train Jaccard/IOU")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.savefig(str(ouput_dir / 'jaccard_iou_score.png'))

    metrics.remove("jaccard_score")
    metrics.remove("iou_score")

    # DICE/FBeta
    fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, hist_metrics[f"DICE{beta}_score"], label="SKLearn_DICE_score")
    ax.plot(epochs, hist_metrics[f"f{beta}-score"], label="SegLib_fbeta_score")
    ax.set(xlabel='Epochs', ylabel=f"F{beta} Score", title=f"Train f{beta}")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.savefig(str(ouput_dir / f"f{beta}_score.png"))
    
    metrics.remove(f"DICE{beta}_score")
    metrics.remove(f"f{beta}-score")

    for metric, vals in hist_metrics.items():
        title = f"Val {metric}" if "val" in metric else f"Train {metric}"
        fig, ax = plt.subplots(1, 1)
        ax.plot(vals)
        ax.set(xlabel='Epoch', ylabel=metric, title=f"Train {metric}")
        ax.grid(True)
        fig.savefig(str(ouput_dir / f'{metric}.png'))
