# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import os
import collections
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, average_precision_score, roc_auc_score, \
    confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tabulate import tabulate
from scipy.special import softmax

# NOTE: This code is primarily written by Neil Joshi. It was modified to work with multiclass data and to avoid using keras/tensorflow unnecessarily.


def evaluate_performance_metrics(y_true, y_pred, output_dir=None, save_roc_curve=True, save_latex_metrics=True,
                                 save_json_metrics=True, save_csv_metrics=True, do_print=False,
                                 normalize_confusion_matrix=True, score_decision_threshold=0.5, decimal_places=4,
                                 z=1.96):
    """Convoluted function to evaluate classification performance and generate metrics in various formats

    :param y_true: either:
                    1) ndarray, shape=(n,) of true class labels
                    2) For k-fold cross validation, a k-length list of ndarrays, shape=(~n/k,) of true class labels
    :param y_pred: either:
                    1a) ndarray, shape=(n,) of predicted class labels
                    1b) ndarray, shape=(n, num_classes) of predicted scores for each class
                    2a) For k-fold cross validation, a k-length list of ndarrays, shape=(~n/k,)
                            of predicted class labels
                    2b) For k-fold cross validation, a k-length list of ndarrays, shape=(~n/k, num_classes)
                            of predicted scores for each class
    :param output_dir: If given, a path to a directory to output metrics in various formats
    :param save_roc_curve: Whether or not to generate and save a ROC curve plot
    :param save_latex_metrics: Whether or not to generate and save metrics in latex-style (for papers)
    :param save_json_metrics: Whether or not to generate and save metrics in json-style (for re-loading later)
    :param save_csv_metrics: Whether or not to generate and save metrics in csv-style (for general purpose)
    :param do_print: Whether or not to print results to console
    :param normalize_confusion_matrix: Whether or not to normalize rows of the confusion matrix
    :param score_decision_threshold: float in the range [0.0, 1.0], threshold of score for positive prediction
    :param decimal_places: int, number of decimal places in metrics
    :param z: numerical value used for confidence interval calculation
    :return: dictionary of metrics and confusion matrix
    """
    # make output directory if necessary
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # handle case of results from k fold (y_true is a list of k elements)
    if type(y_true) is list:
        num_k_folds = len(y_true)

        bincount = np.bincount(np.concatenate(y_true))
        num_classes = len(bincount)
        if do_print:
            print("\n----- K=" + str(num_k_folds) + " FOLD STATISTICS -----\n")
            print("Class Balancing (Across All Data):", bincount)
    else:
        num_k_folds = 1
        y_true = [y_true]
        y_pred = [y_pred]

        bincount = np.bincount(np.concatenate(y_true))
        num_classes = len(bincount)
        if do_print:
            print("\n----- TEST SET STATISTICS -----\n")
            print("Class Balancing (Test Set):", bincount)

    # y_pred can be either class predictions labels (1D) or prediction scores for each class (2D)
    if y_pred[0].ndim == 2:
        # if it is prediction scores, y_pred is actually y_scores, and we must generate the actual y_pred manually
        y_score = y_pred
        if num_classes == 2:
            y_pred = [(y_score[i][:, 1] >= score_decision_threshold).astype(np.int64) for i in range(len(y_score))]
        else:
            y_pred = [np.argmax(y_score[i], axis=1) for i in range(len(y_score))]
    else:
        # otherwise, we have no prediction scores (certain metrics will not be calculated as a result)
        y_score = None

    metrics_dict = collections.OrderedDict()
    if num_classes == 2:
        metrics_list = ["Accuracy", "Sensitivity/Recall", "Specificity", "PPV/Precision", "NPV", "Unweighted Kappa", "Weighted Kappa",
                        "Positive Likelihood", "Negative Likelihood", "F1 Score", "Average Precision", "AUC"]
    else:
        metrics_list = ["Accuracy", "Sensitivity/Recall", "Specificity", "PPV/Precision", "NPV", "Unweighted Kappa", "Weighted Kappa",
                        "F1 Score", "Average Precision", "AUC"]

    for metric in metrics_list:
        metrics_dict[metric] = []

    for k in range(num_k_folds):
        if y_score is not None and len(y_score[k].shape) == 1:
            y_score[k] = np.column_stack((-1 * (y_score[k] - 1), y_score[k]))

        metrics_dict["Accuracy"].append(
            compute_metric_with_conf_interval(metric_score=accuracy_score(y_true[k], y_pred[k]), n=len(y_true[k]), z=z,
                                              is_percentage=True, decimal_places=decimal_places - 2))
        metrics_dict["Unweighted Kappa"].append(
            (round(cohen_kappa_score(y_true[k], y_pred[k], weights=None), decimal_places), -1))
        metrics_dict["Weighted Kappa"].append(
            (round(cohen_kappa_score(y_true[k], y_pred[k], weights='linear'), decimal_places), -1))

        if num_classes == 2:
            cm = confusion_matrix(y_true[k], y_pred[k], labels=[0, 1])
            TN = cm[0][0]
            FP = cm[0][1]
            FN = cm[1][0]
            TP = cm[1][1]

            metrics_dict["Sensitivity/Recall"].append(
                compute_metric_with_conf_interval(metric_score=(TP * 1.0) / (TP + FN), n=(TP + FN), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["Specificity"].append(
                compute_metric_with_conf_interval(metric_score=(TN * 1.0) / (FP + TN), n=(FP + TN), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["PPV/Precision"].append(
                compute_metric_with_conf_interval(metric_score=(TP * 1.0) / (TP + FP), n=(TP + FP), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["NPV"].append(
                compute_metric_with_conf_interval(metric_score=(TN * 1.0) / (TN + FN), n=(TN + FN), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["Positive Likelihood"].append(
                (round(((TP * 1.0) / (TP + FN)) / (1 - ((TN * 1.0) / (FP + TN))), decimal_places), -1))
            metrics_dict["Negative Likelihood"].append(
                (round((1 - ((TP * 1.0) / (TP + FN))) / ((TN * 1.0) / (FP + TN)), decimal_places), -1))
            metrics_dict["F1 Score"].append((round(f1_score(y_true[k], y_pred[k]), decimal_places), -1))

            if y_score is not None and len(np.unique(y_true)) != 1:
                metrics_dict["Average Precision"].append(
                    compute_metric_with_conf_interval(metric_score=average_precision_score(y_true[k], y_score[k][:, 1]),
                                                      n=len(y_true[k]), z=z, decimal_places=decimal_places))
                metrics_dict["AUC"].append(
                    compute_metric_with_conf_interval(metric_score=roc_auc_score(y_true[k], y_score[k][:, 1]),
                                                      n=len(y_true[k]), z=z, decimal_places=decimal_places))
            else:
                metrics_dict["Average Precision"].append((-1, -1))
                metrics_dict["AUC"].append((-1, -1))
        else:
            _, multiclass_sensitivity, multiclass_specificity, multiclass_precision, multiclass_npv, _, _, _ = _performance_multilabel(
                y_true[k], y_pred[k],
                y_score=y_score[k] if y_score is not None and len(np.unique(y_true)) != 1 else None)

            metrics_dict["Sensitivity/Recall"].append(
                compute_metric_with_conf_interval(metric_score=multiclass_sensitivity, n=len(y_true[k]), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["Specificity"].append(
                compute_metric_with_conf_interval(metric_score=multiclass_specificity, n=len(y_true[k]), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["PPV/Precision"].append(
                compute_metric_with_conf_interval(metric_score=multiclass_precision, n=len(y_true[k]), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["NPV"].append(
                compute_metric_with_conf_interval(metric_score=multiclass_npv, n=len(y_true[k]), z=z,
                                                  is_percentage=True, decimal_places=decimal_places - 2))
            metrics_dict["F1 Score"].append(
                (round(f1_score(y_true[k], y_pred[k], average='micro'), decimal_places), -1))

            if y_score is not None and len(np.unique(y_true)) != 1:
                metrics_dict["Average Precision"].append(
                    compute_metric_with_conf_interval(metric_score=average_precision_score(label_binarize(y_true[k], classes=np.array([i for i in range(num_classes)])), y_score[k], average='micro'),
                                                      n=len(y_true[k]), z=z, decimal_places=decimal_places))

            if y_score is not None and len(np.unique(y_true)) != 1:
                if num_classes > 2:
                    # for the multiclass case, y_true needs to have at least 1 sample from every class
                    y_true_modified = y_true[k].tolist()
                    y_score_modified = y_score[k]
                    for j in range(num_classes):
                        if not (y_true[k] == j).max(): # check if at least once instance of the given class does not exist
                            y_true_modified.append(j)
                            filler = np.zeros([1, num_classes])
                            idx = j + 1 if j + 1 < num_classes else j -1 # intentionally guess incorrectly
                            filler[0, idx] = 1.0
                            y_score_modified = np.vstack((y_score_modified, filler))

                    y_true_modified = np.array(y_true_modified)
                    y_score_prob = softmax(y_score_modified, axis=1)
                    metrics_dict["AUC"].append(
                        compute_metric_with_conf_interval(metric_score=roc_auc_score(y_true_modified, y_score_prob, average='macro', multi_class='ovo'), n=len(y_true_modified), z=z,
                                                        decimal_places=decimal_places))
                else:
                    metrics_dict["AUC"].append(
                        compute_metric_with_conf_interval(metric_score=roc_auc_score(y_true[k], y_score[k], average='macro', multi_class='ovo'), n=len(y_true[k]), z=z,
                                                        decimal_places=decimal_places))
            else:
                metrics_dict["AUC"].append((-1, -1))

    np.set_printoptions(precision=decimal_places)

    stats_metric = np.zeros(len(metrics_list), dtype=object)
    stats_metric_std = np.zeros(len(metrics_list), dtype=object)
    for i, metric in enumerate(metrics_list):
        stats_metric[i], stats_metric_std[i] = np.mean(metrics_dict[metric], axis=0).tolist()
        # currently for k fold metrics, we use the std dev of the metrics across all k folds instead of 95% CI
        if num_k_folds > 1:
            stats_metric_std[i] = np.std([num[0] for num in metrics_dict[metric]], ddof=1)

    if num_k_folds > 1:
        stats_table = np.column_stack((stats_metric, stats_metric_std))
        col_labels = np.asarray(['val', 'std'])
    else:
        stats_table = np.column_stack((stats_metric, stats_metric_std))
        col_labels = np.asarray(['val', 'CI'])
    row_labels = metrics_list
    arr = np.column_stack((row_labels, stats_table.astype(dtype='object')))

    if do_print:
        print('\n' + tabulate(arr, headers=col_labels, tablefmt='simple'))

    if save_latex_metrics and output_dir is not None:
        with open(os.path.join(output_dir, "latex_metrics.tex"), 'w') as fp:
            fp.write(tabulate(arr, headers=col_labels, tablefmt='latex'))

    cm = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred))

    if normalize_confusion_matrix:
        cm = np.asarray(cm, dtype=np.float32)
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.around(cm, decimals=decimal_places)
    metrics_dict['Confusion Matrix'] = str(cm)

    if do_print:
        if normalize_confusion_matrix:
            print("\nConfusion Matrix (in %):")
        else:
            print("\nConfusion Matrix:")
        print(cm)

    if num_k_folds == 1:
        cr = classification_report(y_true[0], y_pred[0], digits=decimal_places)

        if do_print:
            print("\nClassification Report:")
            print(cr)

        metrics_dict['Classification Report'] = str(cr)

    if save_roc_curve and output_dir is not None and num_classes == 2:
        fpr, tpr, thresholds = roc_curve(np.concatenate(y_true), np.concatenate(y_score)[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        old_backend = matplotlib.get_backend()
        plt.switch_backend('agg')
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "ROC_curve.pdf"), bbox_inches='tight')
        plt.switch_backend(old_backend)

    if save_json_metrics and output_dir is not None:
        with open(os.path.join(output_dir, "json_metrics.json"), 'w') as fp:
            j = json.dumps(metrics_dict, indent=4)
            fp.write(j)

    if save_csv_metrics and output_dir is not None:
        metrics_dict_for_csv = metrics_dict.copy()
        for key in metrics_dict.keys():
            if isinstance(metrics_dict_for_csv[key], list):
                metrics_dict_for_csv[key] = [
                    "{0} ({1})".format(*metrics_dict_for_csv[key][0]) if metrics_dict_for_csv[key][0][1] != -1
                    else "{0}".format(metrics_dict_for_csv[key][0][0])]
            elif isinstance(metrics_dict_for_csv[key], str):
                metrics_dict_for_csv[key] = [metrics_dict_for_csv[key]]
            else:
                metrics_dict_for_csv.pop(key)

        metrics_df = pd.DataFrame(metrics_dict_for_csv)
        metrics_df = metrics_df.round(4)

        metrics_df.to_csv(os.path.join(output_dir, "csv_metrics.csv"), index=False)

    return metrics_dict, cm


def compute_metric_with_conf_interval(metric_score, n, z, is_percentage=False, decimal_places=None):
    mult = 100.0 if is_percentage else 1.0
    if decimal_places is None:
        decimal_places = 2 if is_percentage else 4
    return round(mult * metric_score, decimal_places), round(
        mult * z * np.sqrt((metric_score * (1.0 - metric_score)) / n), decimal_places)

def _performance_multilabel(y_truth, y_prediction, y_score=None, beta=1):
    '''
    Multiclass performance metrics. (adapted from WORC library)

    y_truth and y_prediction should both be lists with the multiclass label of each
    object, e.g.

    y_truth = [0, 0,	0,	0,	0,	0,	2,	2,	1,	1,	2]    ### Groundtruth
    y_prediction = [0, 0,	0,	0,	0,	0,	1,	2,	1,	2,	2]    ### Predicted labels
    y_score = [[0.3, 0.3, 0.4], [0.2, 0.6, 0.2], ... ] # Normalized score per patient for all labels (three in this example)


    Calculation of accuracy accorading to formula suggested in CAD Dementia Grand Challege http://caddementia.grand-challenge.org
    and the TADPOLE challenge https://tadpole.grand-challenge.org/Performance_Metrics/
    Calculation of Multi Class AUC according to classpy: https://bitbucket.org/bigr_erasmusmc/classpy/src/master/classpy/multi_class_auc.py

    '''
    cm = confusion_matrix(y_truth, y_prediction)

    # Determine no. of classes
    labels_class = np.unique(y_truth)
    n_class = len(labels_class)

    # Splits confusion matrix in true and false positives and negatives
    TP = np.zeros(shape=(1, n_class), dtype=int)
    FN = np.zeros(shape=(1, n_class), dtype=int)
    FP = np.zeros(shape=(1, n_class), dtype=int)
    TN = np.zeros(shape=(1, n_class), dtype=int)
    n = np.zeros(shape=(1, n_class), dtype=int)
    for i in range(n_class):
        TP[:, i] = cm[i, i]
        FN[:, i] = np.sum(cm[i, :]) - cm[i, i]
        FP[:, i] = np.sum(cm[:, i]) - cm[i, i]
        TN[:, i] = np.sum(cm[:]) - TP[:, i] - FP[:, i] - FN[:, i]

    n = np.sum(cm)

    # Determine Accuracy
    Accuracy = (np.sum(TP)) / n

    # BCA: Balanced Class Accuracy
    BCA = list()
    for i in range(n_class):
        BCAi = 1 / 2 * (TP[:, i] / (TP[:, i] + FN[:, i]) + TN[:, i] / (TN[:, i] + FP[:, i]))
        BCA.append(BCAi)

    AverageAccuracy = np.mean(BCA)

    # Determine total positives and negatives
    P = TP + FN
    N = FP + TN

    # Calculation of sensitivity
    Sensitivity = TP / P
    Sensitivity = np.mean(Sensitivity)

    # Calculation of specifitity
    Specificity = TN / N
    Specificity = np.mean(Specificity)

    # Calculation of precision
    Precision = TP / (TP + FP)
    Precision = np.nan_to_num(Precision)
    Precision = np.mean(Precision)

    # Calculation of NPV
    NPV = TN / (TN + FN)
    NPV = np.nan_to_num(NPV)
    NPV = np.mean(NPV)

    # Calculation  of F1_Score
    F1_score = ((1 + (beta ** 2)) * (Sensitivity * Precision)) / ((beta ** 2) * (Precision + Sensitivity))
    F1_score = np.nan_to_num(F1_score)
    F1_score = np.mean(F1_score)

    # Calculation of Multi Class AUC according to classpy: https://bitbucket.org/bigr_erasmusmc/classpy/src/master/classpy/multi_class_auc.py
    if y_score is not None:
        AUC = _multi_class_auc(y_truth, y_score)
    else:
        AUC = None

    return Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy

def _multi_class_auc(y_truth, y_score):
    """
    Adapted from WORC library
    """
    classes = np.unique(y_truth)

    # if any(t == 0.0 for t in np.sum(y_score, axis=1)):
    #     raise ValueError('No AUC is calculated, output probabilities are missing')

    pairwise_auc_list = [0.5 * (_pairwise_auc(y_truth, y_score, i, j) + _pairwise_auc(y_truth, y_score, j, i)) for i in classes for j in classes if i < j]

    c = len(classes)
    return (2.0 * sum(pairwise_auc_list)) / (c * (c - 1))

def _pairwise_auc(y_truth, y_score, class_i, class_j):
    """
    Adapted from WORC library
    """
    # Filter out the probabilities for class_i and class_j
    y_score = [est[class_i] for ref, est in zip(y_truth, y_score) if ref in (class_i, class_j)]
    y_truth = [ref for ref in y_truth if ref in (class_i, class_j)]

    # Sort the y_truth by the estimated probabilities
    sorted_y_truth = [y for x, y in sorted(zip(y_score, y_truth), key=lambda p: p[0])]

    # Calculated the sum of ranks for class_i
    sum_rank = 0
    for index, element in enumerate(sorted_y_truth):
        if element == class_i:
            sum_rank += index + 1
    sum_rank = float(sum_rank)

    # Get the counts for class_i and class_j
    n_class_i = float(y_truth.count(class_i))
    n_class_j = float(y_truth.count(class_j))

    # If a class in empty, AUC is 0.0
    if n_class_i == 0 or n_class_j == 0:
        return 0.0

    # Calculate the pairwise AUC
    return (sum_rank - (0.5 * n_class_i * (n_class_i + 1))) / (n_class_i * n_class_j)
