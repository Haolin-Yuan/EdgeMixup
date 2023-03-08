# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import time
import multiprocessing

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
from segmentation_models.metrics import IOUScore
from sklearn.utils import shuffle

from config import *
from skin_segmentation.segmentation_cfg import *
from skin_segmentation.model_utils import *
from skin_segmentation.data_utils import DataLoader, Dataset
from skin_segmentation.plot_utils import plot_images_w_masks


def load_predictions(data_set, set_name, model_path, batch_size, image_size):
    preds_arr = None
    cache_works = True
    if set_name not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid set name, must be either \"train\", \"val\", or \"test\" (not {set_name}) ")
    else:
        cache_path = OUTPUT_DIR / f'softmax_preds_{set_name}.npz'
    
    while preds_arr is None:  # computes the prediction masks, or loads them if they're already saved
        if cache_path.exists() and cache_works:
            print("Feature cache found, now loading...")

            preds_arr = np.load(str(cache_path), allow_pickle=True)['preds_arr']

            if len(data_set) == preds_arr.shape[0]:
                print("Feature cache loaded \n")
            else:
                preds_arr = None
                cache_works = not cache_works
                print("Features do not correspond to the current data, they will now be re-computed \n")
        else:
            print("Extracting features...")

            s = time.time()
            _, preds_arr = predict_mask_generator(data_set.im_paths, model_path, image_size, batch_size, make_masks_usable=False)
            np.savez(str(cache_path), preds_arr=preds_arr)

            print(f"Features extracted and saved in {time.time() - s} seconds \n")
    
    return preds_arr


def generate_data(data_set, set_name, model_path, batch_size, image_size, T_min=.5, T_max=1, T_step=.05):
    preds_arr = load_predictions(data_set, set_name, model_path, batch_size, image_size)

    T_range = np.arange(T_min, T_max, T_step)
    mean_ious = []
    for i, T in enumerate(T_range):
        pr_masks = bounded_decision_function(preds_arr, T=T)

        ious = []
        for j in range(len(data_set)):
            iou = IOUScore(threshold=0.5)(data_set[j][1], keras.utils.to_categorical(pr_masks[j]))
            with keras.backend.get_session() as sess:
                iou = sess.run(iou)
            keras.backend.clear_session()

            ious.append(iou)

        mean_ious.append(np.mean(ious))

    return {"T": T_range, "IOU": mean_ious}


def compute_mean_IOU(data_set, set_name, T, model_path, batch_size, image_size):
    preds_arr = load_predictions(data_set, set_name, model_path, batch_size, image_size)
    pr_masks = bounded_decision_function(preds_arr, T=T)

    ious = []
    for j in range(len(data_set)):
        iou = IOUScore(threshold=0.5)(data_set[j][1], keras.utils.to_categorical(pr_masks[j]))
        with keras.backend.get_session() as sess:
            iou = sess.run(iou)
        keras.backend.clear_session()

        ious.append(iou)
    
    return np.mean(ious)


def plot_data(val_data_dict, T_best, test_mean_iou):
    plt.plot(val_data_dict["T"], val_data_dict["IOU"])
    plt.xlabel("T")
    plt.ylabel("IOU (On Validation)")
    plt.suptitle(f"Max at T = {T_best:.2f} (Test IOU = {test_mean_iou})")
    plt.savefig(str(OUTPUT_DIR / "T_vs_IOU_plot.png"))


def main():
    val_sets, test_sets = [], []
    for img_root_dir, mask_json_dir in zip(IMG_ROOT_DIRS, MASK_JSON_DIRS):
        _, val_set, test_set = create_train_val_and_test_sets(img_root_dir, mask_json_dir, TRAIN_SIZE, IMAGE_SIZE)
        val_sets.append(val_set)
        test_sets.append(test_set)
    
    val_set = Dataset([], None, image_size=IMAGE_SIZE, augmentations=val_sets[0].aug_funcs, preprocessing=val_sets[0].preproc_func)
    for dataset in val_sets:
        val_set += dataset
    
    test_set = Dataset([], None, image_size=IMAGE_SIZE, augmentations=test_sets[0].aug_funcs, preprocessing=test_sets[0].preproc_func)
    for dataset in test_sets:
        test_set += dataset

    val_data_dict = generate_data(val_set, 'val', MODEL_PATH, BATCH_SIZE, IMAGE_SIZE, T_min=0.5, T_max=1, T_step=0.05)
    T_best = val_data_dict['T'][np.argmax(val_data_dict['IOU'])]
    
    test_mean_iou = compute_mean_IOU(test_set, 'test', T_best, MODEL_PATH, BATCH_SIZE, IMAGE_SIZE)

    plot_data(val_data_dict, T_best, test_mean_iou)
    plot_images_w_masks(test_set, MODEL_PATH, OUTPUT_DIR, T=T_best, num_plots='all', save_plots=True)


if __name__ == '__main__':
    main()
