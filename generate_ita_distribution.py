# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config as cfg
from ita_data.define_ita_bins import get_ita_data


def plot_ITA_distribution(skin_tone_csv, dataset_csvs=None, labels=None, save_path=None, show_plot=False):
    skin_tone_df = pd.read_csv(str(skin_tone_csv))

    if labels is None:
        bar_labels = np.unique(skin_tone_df['category'])
    else:
        bar_labels = labels

    if dataset_csvs is not None:
        reference_dfs = {}
        for csv_path in dataset_csvs:
            reference_dfs[csv_path.stem] = pd.read_csv(str(csv_path))

        counts_by_df = np.zeros(shape=(len(reference_dfs.keys()), len(bar_labels)), dtype=int)
        for i, label in enumerate(bar_labels):
            ims_w_curr_label = list(skin_tone_df[skin_tone_df['category'] == label]['image'])
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

        vals, counts = np.unique(skin_tone_df['category'], return_counts=True)
        counts = [counts[list(vals).index(label)] if label in vals else 0 for label in bar_labels ]
        
        ax.bar(bar_labels, counts)
        ax.set_title('Distribution of ITA Categories in Dataset')
        ax.set_xlabel("ITA Categories")
        for i, v in enumerate(counts):
            v_offset = max(counts) // 100
            ax.text(i, v + v_offset, str(v), color='black', fontweight='bold', ha='center')

    if save_path is not None:
        fig.savefig(str(save_path))

    if show_plot:
        plt.show()

    plt.close()


def main():
    labels = list(get_ita_data().keys())
    save_path = 'lyme_segmentation/ita_distribution.png'
    # dataset_csvs = []
    # for csv in (TRAIN_CSV, VAL_CSV, TEST_CSV):
    #     if csv is not None:
    #         dataset_csvs.append(csv)

    plot_ITA_distribution(SKIN_TONE_CSV, dataset_csvs=None,
                          labels=labels, save_path=save_path, show_plot=True)


if __name__ == '__main__':
    main()
