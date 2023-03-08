# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import sys
from pathlib import Path
sys.path.append("..")
import multiprocessing

import cv2
import numpy as np
import pandas as pd
import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import segmentation_models as sm
from lyme_segmentation.config import Label
import skin_segmentation.segmentation_cfg as cfg
from skin_segmentation.data_utils import Dataset, DataLoader, randomly_flip_image, ITADataLoader
from skin_segmentation.metrics_utils import JaccardScore, DICEScore
from skin_segmentation.losses import categorical_focal_loss
from sklearn.metrics import jaccard_score, fbeta_score

import pdb

def loss_function(num_classes=3):
    def loss(y_true, y_pred):
        dice_loss = sm.losses.DiceLoss(class_weights=cfg.DICE_LOSS_WEIGHTS)(y_true, y_pred) # TODO: play with these coefficients be careful with these coefficients
        #focal_loss = sm.losses.BinaryFocalLoss()(y_true, y_pred) if num_classes == 1 else sm.losses.CategoricalFocalLoss()(y_true, y_pred) #use different implementation with alpha = tensor of size 3, simialr weighting to dice
        focal_loss = sm.losses.BinaryFocalLoss()(y_true, y_pred) if num_classes == 1 else categorical_focal_loss(alpha=cfg.FOCAL_LOSS_WEIGHTS)(y_true, y_pred) #use different implementation with alpha = tensor of size 3, simialr weighting to dice
        total_loss = dice_loss + (cfg.SEG_LOSS_BETA * focal_loss) # TODO: play with the beta
        
        return total_loss
    return loss

# DISCLAIMER This does not work with the existing code, it can be used as a guide if you want to implement kfold
def k_fold_split(k, img_root_dir, mask_json_dir, image_size):
    check_image_dimensions(image_size)

    im_paths = list(img_root_dir.iterdir())
    im_paths = shuffle(im_paths, random_state=cfg.RANDOM_STATE)

    folds = [im_paths[i * (len(im_paths) // k):(i + 1) * (len(im_paths) // k)] for i in range(k)] # TODO Figure out how to make the folds more equal in size
    
    splits = []
    for i in range(len(folds)):
        val_fold_num = i % k
        test_fold_num = (i + 1) % k
        val_ims, test_ims = folds[val_fold_num], folds[test_fold_num]
        train_ims = [p for j, fold in enumerate(folds) for p in fold if j != val_fold_num and j != test_fold_num]

        train_set = Dataset(im_paths=train_ims, image_size=cfg.IMAGE_SIZE, mask_json_dir=mask_json_dir, augmentations=[randomly_flip_image])
        val_set = Dataset(im_paths=val_ims, image_size=cfg.IMAGE_SIZE, mask_json_dir=mask_json_dir, augmentations=None)
        test_set = Dataset(im_paths=test_ims, image_size=cfg.IMAGE_SIZE, mask_json_dir=mask_json_dir, augmentations=None)

        splits.append({'train': train_set, 'val': val_set, 'test': test_set})
   
    return splits


def create_model(backbone, model_path, num_classes=3, lr=.001):
    callbacks = [ModelCheckpoint(str(model_path), save_best_only=True, mode='min')]
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=cfg.STOPPING_PATIENCE) # have patience work on mean jaccard of the val_loss or the f1
    callbacks.append(early_stopping_callback)
    model = sm.Unet(backbone, classes=num_classes, activation='softmax', encoder_weights='imagenet')
    optim = keras.optimizers.Adam(lr)
    #metrics = [JaccardScore(), DICEScore()]
    metrics = [JaccardScore(), DICEScore(), sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    loss = loss_function(num_classes=num_classes)
    model.compile(optim, loss, metrics, run_eagerly=True) # Training in eager execution

    return model, callbacks, metrics 


def train_model(train_set, val_set, batch_size, backbone, epochs, lr, model_path):
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model, callbacks, _ = create_model(backbone, model_path, lr=lr)

    if cfg.RETURN_SUMMARY:
        model.summary()

    if not cfg.OUTPUT_DIR.exists():
        cfg.OUTPUT_DIR.mkdir()

    hist = model.fit_generator(train_loader,
                               steps_per_epoch=len(train_loader),
                               epochs=epochs,
                               callbacks=callbacks,
                               validation_data=val_loader,
                               validation_steps=len(val_loader),
                               use_multiprocessing=False,
                               workers=multiprocessing.cpu_count())

    return hist


def predict_mask(img, model_path):
    model = keras.models.load_model(str(cfg.MODEL_PATH),
                                    custom_objects={'loss': loss_function(),
                                                    'jaccard_score': JaccardScore(),
                                                    f"DICE{int(cfg.METRIC_BETA)}_score": DICEScore(),
                                                    'iou_score':sm.metrics.IOUScore(threshold=0.5),
                                                    f"f{int(cfg.METRIC_BETA)}-score":sm.metrics.FScore(threshold=0.5)})
    model_input = img
    model.run_eagerly = True

    if len(model_input.shape) == 4:
        pred = model.predict(model_input)
    else:
        pred = model.predict(np.expand_dims(model_input, axis=0))
    
    return model_input, pred


def predict_mask_generator(img_set, custom_loader=None, batch_size=cfg.SKIN_SEG_BATCH_SIZE, make_masks_usable=True, return_raw_imgs=True):
    if custom_loader is not None:
        img_loader = custom_loader
    else:
        img_loader = DataLoader(img_set, batch_size=batch_size, shuffle_every_epoch=False)
    
    model = keras.models.load_model(str(cfg.MODEL_PATH),
                                    custom_objects={'loss': loss_function(),
                                                    'jaccard_score': JaccardScore(),
                                                    f"DICE{int(cfg.METRIC_BETA)}_score": DICEScore(),
                                                    'iou_score':sm.metrics.IOUScore(threshold=0.5),
                                                    f"f{int(cfg.METRIC_BETA)}-score":sm.metrics.FScore(threshold=0.5)})
    model.run_eagerly = True
    preds = model.predict_generator(img_loader, use_multiprocessing=True, workers=multiprocessing.cpu_count(), verbose=1, steps=len(img_loader))

    if make_masks_usable:
        return make_mask_usable(preds)
    else:
        return np.asarray(preds)

def test_model(test_set, output_dir, batch_size=cfg.BATCH_SIZE):
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle_every_epoch=False)
    model = keras.models.load_model(str(cfg.MODEL_PATH),
                                    custom_objects={'loss': loss_function(),
                                                    'jaccard_score': JaccardScore(),
                                                    f"DICE{int(cfg.METRIC_BETA)}_score": DICEScore(),
                                                    'iou_score':sm.metrics.IOUScore(threshold=0.5),
                                                    f"f{int(cfg.METRIC_BETA)}-score":sm.metrics.FScore(threshold=0.5)})
    model.run_eagerly = True
    
    scores = model.evaluate(test_loader, use_multiprocessing=True, workers=multiprocessing.cpu_count())
    
    j_scores, dice_scores = [], []
    j_score_conf_intervals, dice_conf_intervals = [], []
    for data in test_loader:
        pred = model.predict(data, use_multiprocessing=True, workers=multiprocessing.cpu_count())
        
        j_scores.append(jaccard_score(
                y_true=data[1].reshape(-1, len(cfg.Label)),
                y_pred=tf.one_hot(tf.argmax(pred, axis=-1), depth=len(cfg.Label)).numpy().reshape(-1, len(cfg.Label)),
                average='macro'))
        j_score_conf_intervals.append(
            round(1.96 * np.sqrt(abs(j_scores[-1] * (1.0 - j_scores[-1])) / (cfg.IMAGE_SIZE[0] * cfg.IMAGE_SIZE[1])), 4)
        )
        dice_scores.append(fbeta_score(
                y_true=data[1].reshape(-1, len(cfg.Label)),
                y_pred=tf.one_hot(tf.argmax(pred, axis=-1), depth=len(cfg.Label)).numpy().reshape(-1, len(cfg.Label)),
                beta=cfg.METRIC_BETA,
                average='macro'))
        dice_conf_intervals.append(
            round(1.96 * np.sqrt(abs(dice_scores[-1] * (1.0 - dice_scores[-1])) / (cfg.IMAGE_SIZE[0] * cfg.IMAGE_SIZE[1])), 4)
        )

    j_score = np.stack(j_scores, axis=0).mean()
    j_score_conf_interval = np.stack(j_score_conf_intervals, axis=0).mean()
    dice_score = np.stack(dice_scores, axis=0).mean()
    dice_conf_interval = np.stack(dice_conf_intervals, axis=0).mean()

    print(f"Jaccard Score {j_score:4} ({j_score_conf_interval}) DICE Score {dice_score:4} ({dice_conf_interval})")

    #j_score_conf_interval = round(1.96 * np.sqrt(abs(j_score * (1.0 - j_score)) / len(j_scores)), 4)
    #dice_conf_interval = round(1.96 * np.sqrt(abs(dice_score * (1.0 - dice_score)) / len(dice_scores)), 4)

    scores = (scores[0], j_score, dice_score, scores[3], scores[4], j_score_conf_interval, dice_conf_interval)
    metrics_names = model.metrics_names
    metrics_names.append('jaccard_score_conf_interval')
    metrics_names.append(f'DICE{int(cfg.METRIC_BETA)}_score_conf_interval')
    
    return scores, metrics_names


def calculate_ITA_test_metrics(test_set, ita_df):
    img_pth2idx = {}
    for i, img_pth in enumerate(ita_df["image"].tolist()):
        img_pth2idx[img_pth] = i
    
    np.random.seed(cfg.RANDOM_STATE)
    tf.random.set_seed(cfg.RANDOM_STATE)
    
    test_loader = ITADataLoader(test_set, preds=None, batch_size=1, return_proc_imgs=True)
    model = keras.models.load_model(str(cfg.MODEL_PATH),
                                    custom_objects={'loss': loss_function(),
                                                    'jaccard_score': JaccardScore(),
                                                    f"DICE{int(cfg.METRIC_BETA)}_score": DICEScore(),
                                                    'iou_score':sm.metrics.IOUScore(threshold=0.5),
                                                    f"f{int(cfg.METRIC_BETA)}-score":sm.metrics.FScore(threshold=0.5)})
    model.run_eagerly = True
    
    all_metrics = {
        'ls' : {
                'Image names': [],
                'GT ITAs': [],
                'Jaccard Scores': [],
                'Jaccard Scores Gap': [],
                'DICE Scores': [],
                'DICE Scores Gap': []
            },
        'ds' : {
                'Image names': [],
                'GT ITAs': [],
                'Jaccard Scores': [],
                'Jaccard Scores Gap': [],
                'DICE Scores': [],
                'DICE Scores Gap': []
            }
    }

    for content in test_loader:
        img_path = Path(content[2][0]).name
        idx = img_pth2idx[img_path]
        
        binary_skintone = 'ds' if ita_df.iloc[idx]['gt_category'] in ['tan2', 'tan1', 'dark'] else 'ls'
        
        all_metrics[binary_skintone]['Image names'].append(img_path)
        all_metrics[binary_skintone]['GT ITAs'].append(ita_df.iloc[idx]['gt_ita'])

        data = (content[0], content[3])
        pred = model.predict(data, use_multiprocessing=False, workers=0)
        
        all_metrics[binary_skintone]['Jaccard Scores'].append(jaccard_score(
                y_true=data[1].reshape(-1, len(cfg.Label)),
                y_pred=tf.one_hot(tf.argmax(pred, axis=-1), depth=len(cfg.Label)).numpy().reshape(-1, len(cfg.Label)),
                average='macro'))
        all_metrics[binary_skintone]['Jaccard Scores Gap'].append(
                round(1.96 * np.sqrt(abs(all_metrics[binary_skintone]['Jaccard Scores'][-1] * (1.0 - all_metrics[binary_skintone]['Jaccard Scores'][-1])) / (cfg.IMAGE_SIZE[0] * cfg.IMAGE_SIZE[1])), 4)
            )
        all_metrics[binary_skintone]['DICE Scores'].append(fbeta_score(
                y_true=data[1].reshape(-1, len(cfg.Label)),
                y_pred=tf.one_hot(tf.argmax(pred, axis=-1), depth=len(cfg.Label)).numpy().reshape(-1, len(cfg.Label)),
                beta=cfg.METRIC_BETA,
                average='macro'))
        all_metrics[binary_skintone]['DICE Scores Gap'].append(
                round(1.96 * np.sqrt(abs(all_metrics[binary_skintone]['DICE Scores'][-1] * (1.0 - all_metrics[binary_skintone]['DICE Scores'][-1])) / (cfg.IMAGE_SIZE[0] * cfg.IMAGE_SIZE[1])), 4)
            )
        print(f"{binary_skintone} Pred: {np.linalg.norm(pred)} ({np.linalg.norm(data[0])}, {np.linalg.norm(data[1])}) Jaccard: {all_metrics[binary_skintone]['Jaccard Scores'][-1]} DICE: {all_metrics[binary_skintone]['DICE Scores'][-1]}")
    
    jaccard_gap = np.array(all_metrics['ls']['Jaccard Scores']).mean() - np.array(all_metrics['ds']['Jaccard Scores']).mean()
    #jaccard_conf_interval = round(1.96 * np.sqrt(abs(jaccard_gap * (1.0 - jaccard_gap)) / len(test_loader)), 4) # For N use the number of pixels per image
    jaccard_gap = round(jaccard_gap, 4)
    jaccard_conf_interval = np.array(all_metrics['ls']['Jaccard Scores Gap']).mean() - np.array(all_metrics['ds']['Jaccard Scores Gap']).mean()
    
    if len(all_metrics['ls']['Jaccard Scores']) > len(all_metrics['ds']['Jaccard Scores']):
        most_common_class = 'ls'
    elif len(all_metrics['ls']['Jaccard Scores']) < len(all_metrics['ds']['Jaccard Scores']):
        most_common_class = 'ds'
    else:
        most_common_class = 'same'
    
    DICE_gap = np.array(all_metrics['ls']['DICE Scores']).mean() - np.array(all_metrics['ds']['DICE Scores']).mean()
    #DICE_conf_interval = round(1.96 * np.sqrt(abs(DICE_gap * (1.0 - DICE_gap)) / len(test_loader)), 4)
    DICE_gap = round(DICE_gap, 4)
    DICE_conf_interval = np.array(all_metrics['ls']['DICE Scores Gap']).mean() - np.array(all_metrics['ds']['DICE Scores Gap']).mean()
    
    print(f"Jaccard gap: {jaccard_gap} ({jaccard_conf_interval}) DICE gap: {DICE_gap} ({DICE_conf_interval}) Most common class: {most_common_class}")
    return all_metrics

def make_mask_usable(model_pred):
    mask = np.argmax(model_pred, axis=-1)
    
    if mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)

    mask = np.expand_dims(mask, axis=-1)

    return mask


def split_mask(pr_mask):
    skin_mask = (pr_mask == Label.SKIN.value).astype(int)
    lesion_mask = (pr_mask == Label.LESION.value).astype(int)
    
    return skin_mask, lesion_mask