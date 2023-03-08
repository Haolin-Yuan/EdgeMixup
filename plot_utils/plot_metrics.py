# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confmat(confmat_name="confmat", confmat_dir=None, **kwargs):
    """
    Plots a confusion matrix from given data
    :param confmat_name: name to give saved confmat
    :param confmat_dir: where to save the confmat to
    :param kwargs: used to specify the data to create the confmat from. If the true and predicted labels are passed in
    (as 'true_labels', 'pred_labels', respectively) then those are used to compute a confmat. If a precomputed confusion
    matrix is passed in (as 'cm'), it is simply annotated and turned into a heatmap.
    :raises RuntimeError: in the case that no data is passed in
    """
    fig2, ax = plt.subplots(1, 1, num=2)

    if 'true_labels' in kwargs and 'pred_labels' in kwargs:
        true_labels, pred_labels = kwargs['true_labels'], kwargs['pred_labels']
        cm = confusion_matrix(true_labels, pred_labels)
    elif 'cm' in kwargs:
        cm = kwargs['cm']
    else:
        raise RuntimeError("No labels or matrix specified")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix
    for pair in np.argwhere(np.isnan(cm_norm)):
        cm_norm[pair[0]][pair[1]] = 0

    annot = np.zeros_like(cm, dtype=object)
    for i in range(annot.shape[0]):  # Creates an annotation array for the heatmap
        for j in range(annot.shape[1]):
            annot[i][j] = f'{cm[i][j]}\n{round(cm_norm[i][j] * 100, ndigits=3)}%'

    ax = sns.heatmap(cm_norm, annot=annot, fmt='', cbar=True, cmap=plt.cm.magma, vmin=0, ax=ax) # plot the confusion matrix

    ax.set(xlabel='Predicted Label', ylabel='Actual Label')

    fig2.tight_layout()
    fig2.savefig(str(confmat_dir / f'{confmat_name}.png'))  # save the confusion matrix
    fig2.clear()