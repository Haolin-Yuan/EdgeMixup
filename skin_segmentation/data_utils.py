# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
import os.path
import sys
sys.path.append("..")
from pathlib import Path
import random
import json
import zlib
import base64
from math import ceil, pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import albumentations
from sklearn.utils import shuffle
from tqdm import tqdm
import skin_segmentation.segmentation_cfg as cfg 

from ita_data.define_ita_bins import get_ita_data
from edge_mixup_util import *

import pdb


def check_image_dimensions(image_size):
    if image_size[0] % 32 != 0 or image_size[1] % 32 != 0: # ensures dimensions are usable
        aspect_ratio = float(max(image_size)) / min(image_size)
        new_min = ((min(image_size) // 32) + 1) * 32
        new_max = new_min * aspect_ratio
        new_dims = (int(new_min), int(new_max)) if image_size[0] == min(image_size) else (int(new_max), int(new_min))

        raise ValueError(f"Image size must be a multiple of 32 (suggested size: {new_dims[0]}x{new_dims[1]})")

def randomly_flip_image(image, mask):
    x = random.randint(1, 10)
    if x <= 5:
        return cv2.flip(image, 1), cv2.flip(mask, 1)
    else:
        return image, mask


def base64_2_mask(s): # from https://docs.supervise.ly/data-organization/import-export/supervisely-format
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def clahe(image):
    aug = albumentations.Compose([
        albumentations.CLAHE(p=1.0), # Contrast Adaptive Histogram Equalization only works on uint8
        albumentations.ImageCompression(quality_lower=25, quality_upper=100, p=0.25),
    ], p=1)
    transform = aug(image=image)
    return transform['image']

def im_aug_func(image, mask):
    aug = albumentations.Compose([
        albumentations.CLAHE(p=1.0), # Contrast Adaptive Histogram Equalization only works on uint8
        albumentations.ImageCompression(quality_lower=25, quality_upper=100, p=0.25),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=(-1.0, 1.0), contrast_limit=(-1.0, 1.0), p=0.5),
        albumentations.RGBShift(r_shift_limit=1, g_shift_limit=1, b_shift_limit=1, p=0.5),
        albumentations.RandomResizedCrop(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1], scale=(0.75, 1.0), ratio=(0.9, 1.0), p=0.5),
        albumentations.MultiplicativeNoise(multiplier=[0.99, 1.0], elementwise=True, per_channel=True, p=0.25),
        #albumentations.GaussianBlur(blur_limit=7, p=0.25),
    ], p=1)
    transform = aug(image=image, mask=mask)
    return transform['image'], transform['mask']

def test_im_aug_func(image, mask):
    aug = albumentations.Compose([
        albumentations.CLAHE(p=1.0), # Contrast Adaptive Histogram Equalization only works on uint8
    ], p=1)
    transform = aug(image=image, mask=mask)
    return transform['image'], transform['mask']

def bounded_decision_function(model_pred, threshold=.5):
    mask = np.zeros(shape=model_pred.shape[:-1])
    '''
    for i, mask in enumerate(masks):
    '''
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if (np.argmax(model_pred[row][col]) == cfg.Label.LESION.value and model_pred[row][col][cfg.Label.LESION.value] >= threshold):
                mask[row][col] = cfg.Label.LESION.value
            else:
                model_pred[row][col][cfg.Label.LESION.value] = -1
                mask[row][col] = np.argmax(model_pred[row][col])
    
    mask = np.expand_dims(mask, axis=-1)
    return mask


class Dataset:
    def __init__(self, im_paths, mask_paths=[], mask_json_dirs=cfg.MASK_JSON_DIRS, image_size=(600, 450),
                 augmentations=None, img_preprocessing=None, dataset_type="all", debug=False,
                 return_raw_imgs=False, edgeMixup = False):
        self.im_paths = im_paths
        if len(mask_paths) > 0:
            self.mask_paths = mask_paths
        else:
            self.mask_paths = []
            if mask_json_dirs is not None:
                for im_path in im_paths:
                    self.mask_paths.append(Path(str(im_path.parent).replace('img', 'ann') ) / (im_path.stem + '.png.json'))          

        self.im_w, self.im_h = image_size
        self.aug_funcs = augmentations
        self.img_preproc_funcs = img_preprocessing
        self.dataset_type = dataset_type
        self.debug = debug
        self.return_raw_imgs = return_raw_imgs
        self.edgemixup = edgeMixup

    def __getitem__(self, item_idx):
        img = cv2.cvtColor(cv2.imread(str(self.im_paths[item_idx])), cv2.COLOR_BGR2RGB)
        orig_shape = img.shape
        img = cv2.resize(img, (self.im_w, self.im_h))

        if self.edgemixup:
            img = getHsvMask(img)

        if self.return_raw_imgs:
            raw_img = np.copy(img)
        
        if len(self.mask_paths) != 0:
            mask = self._load_mask(mask_json=self.mask_paths[item_idx], img_size=orig_shape)
        else:
            mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))
        
        if self.debug:
            print(f"img.shape: {img.shape} mask.shape {mask.shape}")

            fig = plt.figure(figsize=(8,8))
            fig.add_subplot(2, 2, 1)
            plt.imshow(img.astype(np.uint8))
            plt.title("Original Image")
            fig.add_subplot(2, 2, 2)
            plt.imshow(mask.astype(np.uint8) * 127)
            plt.title("Original Mask")
        
        # Augment first, then preprocess
        if self.aug_funcs is not None:
            for f in self.aug_funcs:
                img, mask = f(img, mask)
        
        if self.img_preproc_funcs is not None:
            for f in self.img_preproc_funcs:
                img = f(img)
        
        if self.debug:
            fig.add_subplot(2, 2, 3)
            plt.imshow((img * 255).astype(np.uint8))
            plt.title("Augmented Image")
            fig.add_subplot(2, 2, 4)
            plt.imshow(mask.astype(np.uint8) * 127)
            plt.title("Augmented Mask")
            fig.suptitle("Image and Mask Augmentation")
            debug_dir = cfg.OUTPUT_DIR / "debug_plots"
            if not debug_dir.exists():
                debug_dir.mkdir(parents=True)
            fig.savefig(str(debug_dir / f'{self.dataset_type}_debug_img_mask_{item_idx}.png'))
        
        if self.return_raw_imgs:
            return img, mask, raw_img

        return img, mask

    def __len__(self):
        return len(self.im_paths)
    
    def __add__(self, other):
        if other.im_w != self.im_w or other.im_h != self.im_h:
            raise ValueError("Can not add datasets with differing desired image sizes")
        if other.preproc_func != self.preproc_func:
            raise ValueError("Cannot add datasets with different preprocessing functions")
        if not ((not other.aug_funcs and not self.aug_funcs) or (other.aug_funcs and self.aug_funcs)):
            raise ValueError("Cannot add datasets if one has augmentation functions and the other doesn't")
        if (other.aug_funcs and self.aug_funcs) and set(other.aug_funcs) != set(self.aug_funcs):
            raise ValueError("Cannot add datasets with different augmentation functions")

        im_paths = self.im_paths + other.im_paths
        mask_paths = self.mask_paths + other.mask_paths

        combo = Dataset(im_paths, None, image_size=(self.im_w, self.im_h), augmentations=self.aug_funcs, preprocessing=self.preproc_func)
        combo.mask_paths = mask_paths

        return combo

    def _load_mask(self, mask_json, img_size):
        with open(str(mask_json)) as f:
            json_dict = json.load(f)

        mask = np.zeros(img_size[:2])
        for mask_dict in json_dict['objects']:
            bmp_str = mask_dict['bitmap']['data']
            curr_mask = base64_2_mask(bmp_str)

            msk_origin_x, msk_origin_y = mask_dict['bitmap']['origin']
            mask_label = mask_dict['classTitle']

            fill_val = cfg.Label.LESION if mask_label == 'Lesion' else cfg.Label.SKIN
            mask[msk_origin_y:msk_origin_y + curr_mask.shape[0], msk_origin_x:msk_origin_x + curr_mask.shape[1]][curr_mask] = fill_val.value

        mask = cv2.resize(mask, (self.im_w, self.im_h))
        mask = np.expand_dims(mask, axis=-1)
        mask = keras.utils.to_categorical(mask)

        return mask


class DataLoader(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=32, shuffle_every_epoch=False, return_masks=True):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size <= len(dataset) else len(dataset)
        self.shuffle_every_epoch = shuffle_every_epoch
        self.return_masks = return_masks
        self.indices = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, item_idx):
        start, stop = item_idx * self.batch_size, (item_idx + 1) * self.batch_size
        if stop > len(self.dataset):
            stop = len(self.dataset)

        batch_imgs, batch_masks = [], []
        for i in range(start, stop):
            if self.dataset.return_raw_imgs:
                img, mask, _ = self.dataset[i]
            else:
                img, mask = self.dataset[i]
            batch_imgs.append(img)
            batch_masks.append(mask)
        
        if self.return_masks:
            return np.asarray(batch_imgs), np.asarray(batch_masks)
        else:
            return np.asarray(batch_imgs)

    def __len__(self):
        return (len(self.dataset) // self.batch_size) + (1 if len(self.dataset) % self.batch_size > 0 else 0)

    def on_epoch_end(self):
        if self.shuffle_every_epoch:
            self.indices = np.random.permutation(self.indices)

class ITADataLoader(keras.utils.Sequence):
    def __init__(self, dataset, preds, batch_size=32, return_proc_imgs=False, return_mask_path=False):
        self.dataset = dataset
        self.preds = preds
        self.batch_size = batch_size if batch_size <= len(dataset) else len(dataset)
        if self.preds is not None:
            assert len(self.dataset) == len(self.preds), "The dataset and predictions must be of equal size"
        self.dataset.return_raw_imgs = True
        self.return_proc_imgs = return_proc_imgs
        self.return_mask_path = return_mask_path

    def __getitem__(self, item_idx):
        start, stop = item_idx * self.batch_size, (item_idx + 1) * self.batch_size
        if stop > len(self.dataset):
            stop = len(self.dataset)

        batch_imgs, batch_raw_imgs, batch_img_paths, batch_mask_paths, batch_gt_masks, batch_pred_masks = [], [], [], [], [], []
        for i in range(start, stop):
            img, gt_mask, raw_img = self.dataset[i]
            batch_imgs.append(img)
            batch_raw_imgs.append(raw_img)
            batch_img_paths.append(self.dataset.im_paths[i])
            batch_gt_masks.append(gt_mask)
            batch_mask_paths.append(self.dataset.mask_paths[i])
            pred_mask = bounded_decision_function(self.preds[i], threshold=cfg.BOUNDED_DECISION_THRESHOLD) if self.preds is not None else None
            batch_pred_masks.append(pred_mask)
        
        if self.return_proc_imgs:
            if self.return_mask_path:
                return np.asarray(batch_imgs), np.asarray(batch_raw_imgs), batch_img_paths, np.asarray(batch_gt_masks), np.asarray(batch_pred_masks), batch_mask_paths
            else:
                return np.asarray(batch_imgs), np.asarray(batch_raw_imgs), batch_img_paths, np.asarray(batch_gt_masks), np.asarray(batch_pred_masks)
        else:
            if self.return_mask_path:
                return np.asarray(batch_raw_imgs), batch_img_paths, np.asarray(batch_gt_masks), np.asarray(batch_pred_masks), batch_mask_paths
            else:
                return np.asarray(batch_raw_imgs), batch_img_paths, np.asarray(batch_gt_masks), np.asarray(batch_pred_masks)

    def __len__(self):
        return (len(self.dataset) // self.batch_size) + (1 if len(self.dataset) % self.batch_size > 0 else 0)

def calc_batch_ITAs(imgs, masks):          
    # using np.hstack() on the images allows us to perform cv2 operations on the whole batch at once
    img_stack = np.hstack(imgs)

    lab_blurred_stack = cv2.cvtColor(cv2.medianBlur(img_stack.astype(np.uint8), ksize=13),
                                     cv2.COLOR_RGB2LAB)  # blurs images and converts them to CIELab colorspace

    L_vals, b_vals = lab_blurred_stack[:, :, 0], lab_blurred_stack[:, :, 2]  # grab all values in L and b channels
    ita_stack = np.arctan((L_vals - 50) / b_vals) * (180 / pi)  # ITA formula

    # We iterate through each ITA map and apply the corresponding mask to it, thus removing any non-skin pixels.
    ita_vals = [np.mean(ita_arr[masks[j].squeeze() == cfg.Label.SKIN.value])
                for j, ita_arr in enumerate(np.hsplit(ita_stack, indices_or_sections=len(imgs)))]
    return ita_vals

def calculate_all_gt_ita(im_paths, mask_json_dir=cfg.MASK_JSON_DIRS):
    ita_bins_data = get_ita_data()  # loads in the data for each ITA category

    all_set = Dataset(
            im_paths=im_paths,
            image_size=cfg.IMAGE_SIZE,
            img_preprocessing=[preprocess_input],
            augmentations=None,
            dataset_type='all',
            debug=False,
            return_raw_imgs=True)
    
    loader = ITADataLoader(all_set, None, batch_size=cfg.SKIN_SEG_BATCH_SIZE, return_mask_path=True)
    all_data = {
        "images": [], 
        "target_masks": [], 
        "gt_binary_skintone": []
    }

    for content in tqdm(loader, desc="Calculating ITAs"):
        batch_raw_imgs, batch_img_paths, batch_gt_masks, _, batch_mask_paths = content
        batch_gt_masks = np.expand_dims(np.argmax(batch_gt_masks, axis=-1), axis=-1)
        gt_ita_vals = calc_batch_ITAs(batch_raw_imgs, batch_gt_masks)

        # Saves the skin tone category for each image to the dataframe (categories defined in
        # skin_tone_categorization_scheme.txt, which is pulled from Table 1 in the original paper).
        for img_path, mask_path, gt_ita in zip(batch_img_paths, batch_mask_paths, gt_ita_vals):
            pred_tone = 'undef'
            gt_tone = 'undef'
            for k in ita_bins_data.keys():
                if ita_bins_data[k]['check_in_bin'](gt_ita):  # checks which bin the current ITA value falls under
                    gt_tone = k
            
            all_data["images"].append(img_path)
            all_data["target_masks"].append(mask_path)
            binary_skintone = 'ds' if gt_tone in ['tan2', 'tan1', 'dark'] else 'ls'
            all_data["gt_binary_skintone"].append(binary_skintone)
            
    return all_data


def create_train_val_and_test_sets(img_root_dir, mask_json_dir, split_percentages, image_size, paths_only=False):
    check_image_dimensions(image_size)

    #TODO: maybe use sklearn's train/test split for this
    im_paths = list(img_root_dir.iterdir())
    im_paths = shuffle(im_paths, random_state=cfg.RANDOM_STATE)

    all_data = calculate_all_gt_ita(im_paths)
    split_sizes = {'train': 0, 'val': 0, 'test': 0}
    split_sizes['test'] = ceil(split_percentages['test'] * len(im_paths) / 2.) * 2 # Round to even number
    split_sizes['val'] = ceil(split_percentages['val'] * len(im_paths) / 2.) * 2
    split_sizes['train'] = int(len(im_paths)- split_sizes['test'] - split_sizes['val'])

    eval_skintone_cnts = {
        'val': {'ls': split_sizes['val'] // 2, 'ds': split_sizes['val'] // 2},
        'test': {'ls': split_sizes['test'] // 2, 'ds': split_sizes['test'] // 2,}
        }
    
    split_data = { 
        'train': {'images':[], 'target_masks':[], 'gt_binary_skintone': []}, 
        'val': {'images':[], 'target_masks':[], 'gt_binary_skintone': []}, 
        'test': {'images':[], 'target_masks':[], 'gt_binary_skintone': []} 
        }
    
    for i in range(len(all_data['images'])):
        binary_skintone = all_data['gt_binary_skintone'][i]

        if eval_skintone_cnts['test'][binary_skintone] > 0:
            split_data['test']['images'].append(all_data['images'][i])
            split_data['test']['target_masks'].append(all_data['target_masks'][i])
            split_data['test']['gt_binary_skintone'].append(binary_skintone)
            eval_skintone_cnts['test'][binary_skintone] -= 1
        elif eval_skintone_cnts['val'][binary_skintone] > 0:
            split_data['val']['images'].append(all_data['images'][i])
            split_data['val']['target_masks'].append(all_data['target_masks'][i])
            split_data['val']['gt_binary_skintone'].append(binary_skintone)
            eval_skintone_cnts['val'][binary_skintone] -= 1
        else:
            split_data['train']['images'].append(all_data['images'][i])
            split_data['train']['target_masks'].append(all_data['target_masks'][i])
            split_data['train']['gt_binary_skintone'].append(binary_skintone)
    
    '''
    train_ims, test_val_ims = im_paths[0:int(len(im_paths) * train_data_percent)], im_paths[int(len(im_paths) * train_data_percent):]
    val_ims, test_ims = test_val_ims[:len(test_val_ims) // 2], test_val_ims[len(test_val_ims) // 2:]

    train_masks, val_masks, test_masks = [], [], []
    for im_path in train_ims:
        train_masks.append(list(mask_json_dir.glob(im_path.stem + '*'))[0])  
    for im_path in val_ims:
        val_masks.append(list(mask_json_dir.glob(im_path.stem + '*'))[0])  
    for im_path in test_ims:
        test_masks.append(list(mask_json_dir.glob(im_path.stem + '*'))[0])  

    return train_ims, train_masks, val_ims, val_masks, test_ims, test_masks
    '''
    return split_data


def generate_splits():
    
    assert cfg.MASK_JSON_DIRS is not None, f"Target masks are missing in {cfg.MASK_JSON_DIRS}"
    #redo the splits, we want to evaluate on everything
    all_classes_split_data = { 
        'train': {'images':[], 'target_masks':[], 'gt_binary_skintone': []}, 
        'val': {'images':[], 'target_masks':[], 'gt_binary_skintone': []}, 
        'test': {'images':[], 'target_masks':[], 'gt_binary_skintone': []} 
        }
    
    for img_root_dir, mask_json_dir in zip(cfg.IMG_ROOT_DIRS, cfg.MASK_JSON_DIRS):
        '''
        train_ims, train_masks, val_ims, val_masks, test_ims, test_masks = create_train_val_and_test_sets(img_root_dir, mask_json_dir, cfg.SPLIT_PERCENTAGES, cfg.IMAGE_SIZE)
        train_set["images"].extend(train_ims)
        train_set["target_masks"].extend(train_masks)
        val_set["images"].extend(val_ims)
        val_set["target_masks"].extend(val_masks)
        test_set["images"].extend(test_ims)
        test_set["target_masks"].extend(test_masks)
        '''
        split_data = create_train_val_and_test_sets(img_root_dir, mask_json_dir, cfg.SPLIT_PERCENTAGES, cfg.IMAGE_SIZE)
        for split_type in ['train', 'val', 'test']:
            all_classes_split_data[split_type]['images'].extend(split_data[split_type]['images'])
            all_classes_split_data[split_type]['target_masks'].extend(split_data[split_type]['target_masks'])
            all_classes_split_data[split_type]['gt_binary_skintone'].extend(split_data[split_type]['gt_binary_skintone'])

    
    train_df = pd.DataFrame.from_dict(all_classes_split_data['train'])
    val_df = pd.DataFrame.from_dict(all_classes_split_data['val'])
    test_df = pd.DataFrame.from_dict(all_classes_split_data['test'])

    if not cfg.DATA_ASSETS_DIR.exists():
        cfg.DATA_ASSETS_DIR.mkdir(parents=True)
        
    train_df.to_csv(str(cfg.DATA_ASSETS_DIR / 'train.csv'), columns=train_df.columns, index=False)
    val_df.to_csv(str(cfg.DATA_ASSETS_DIR / 'val.csv'), columns=val_df.columns, index=False)
    test_df.to_csv(str(cfg.DATA_ASSETS_DIR / 'test.csv'), columns=test_df.columns, index=False)
    

def load_splits(split=True, return_raw_imgs=False, edgemixup=False):
    train_df = pd.read_csv(str(cfg.DATA_ASSETS_DIR / 'train.csv'))
    val_df = pd.read_csv(str(cfg.DATA_ASSETS_DIR / 'val.csv'))
    test_df = pd.read_csv(str(cfg.DATA_ASSETS_DIR / 'test.csv'))

    if edgemixup and os.path.exists(cfg.MODEL_PATH):
        # start from second iteration
        train_df = add_segment_boundary(train_df, cfg.Iter_root, cfg.MODEL_PATH)
        val_df = add_segment_boundary(val_df, cfg.Iter_root, cfg.MODEL_PATH)
        test_df = add_segment_boundary(test_df, cfg.Iter_root, cfg.MODEL_PATH)


    if not split:
        im_paths = train_df['images'].to_list() + val_df['images'].to_list() + test_df['images'].to_list()
        mask_paths = train_df['target_masks'].to_list() + val_df['target_masks'].to_list() + test_df['target_masks'].to_list()
        all_set = Dataset(
            im_paths=im_paths, 
            mask_paths=mask_paths,
            image_size=cfg.IMAGE_SIZE,
            img_preprocessing=[preprocess_input],
            augmentations=None,
            dataset_type='all',
            debug=False,
            return_raw_imgs=return_raw_imgs,
            edgeMixup = edgemixup,
        )
        #all_pths_df = pd.DataFrame.from_dict({'img_paths': im_paths}) # Not necessary, but sometimes usful
        #all_pths_df.to_csv(str(cfg.OUTPUT_DIR / 'all_im_paths.csv'))
        return all_set

    train_set = Dataset(
        im_paths=train_df['images'].to_list(), 
        mask_paths=train_df['target_masks'].to_list(),
        image_size=cfg.IMAGE_SIZE,
        img_preprocessing=[preprocess_input], # Imagenet normalization
        augmentations=[im_aug_func],
        dataset_type='train',
        debug=False,
        return_raw_imgs=return_raw_imgs,
        edgeMixup=edgemixup,
    )
    
    val_set = Dataset(
        im_paths=val_df['images'].to_list(), 
        mask_paths=val_df['target_masks'].to_list(), 
        image_size=cfg.IMAGE_SIZE,
        img_preprocessing=[preprocess_input],
        augmentations=[], #[test_im_aug_func],
        dataset_type='val',
        return_raw_imgs=return_raw_imgs,
        edgeMixup=edgemixup,
    )

    test_set = Dataset(
        im_paths=test_df['images'].to_list(), 
        mask_paths=test_df['target_masks'].to_list(), 
        image_size=cfg.IMAGE_SIZE,
        img_preprocessing=[preprocess_input],
        augmentations=[],#[test_im_aug_func],
        dataset_type='test',
        return_raw_imgs=return_raw_imgs,
        edgeMixup=edgemixup,
    )
    
    return train_set, val_set, test_set