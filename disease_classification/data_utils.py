# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
import albumentations

import torch as T
from torch.nn.functional import softmax, relu
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from ita_data.define_ita_bins import get_ita_data
from skin_segmentation.data_utils import calc_batch_ITAs, bounded_decision_function, im_aug_func
from skin_segmentation.model_utils import predict_mask_generator
import disease_classification.clf_config as cfg
import sys
sys.append("../")
from edge_mixup_util import *


import pdb

def clf_im_aug_func(image, mask):
    aug = albumentations.Compose([
        #albumentations.CLAHE(p=0.5), # Contrast Adaptive Histogram Equalization only works on uint8
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        albumentations.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        albumentations.RandomResizedCrop(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1], scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.5),
        albumentations.MultiplicativeNoise(multiplier=[0.95, 1.05], elementwise=True, per_channel=True, p=0.25),
        #albumentations.GaussianBlur(blur_limit=7, p=0.25),
        albumentations.ImageCompression(quality_lower=25, quality_upper=100, p=0.25),
    ], p=1.0)
    transform = aug(image=image, mask=mask)
    return transform['image'], transform['mask']

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = T.as_tensor(mean)
        std = T.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

# ////////////////////////////////////////////////
# Raw Data Cleaning
# ////////////////////////////////////////////////
class RawDataset:
    def __init__(self, im_paths, labels, image_size=(600, 450), img_preprocessing=None, dataset_type="all", debug=False):
        '''
        Note: This dataset is used just to read in the raw images to run through the segmentation model and to calculate ITA
        '''
        self.im_paths = im_paths
        self.all_labels = labels        

        self.im_w, self.im_h = image_size
        self.img_preproc_funcs = img_preprocessing
        self.dataset_type = dataset_type
        self.debug = debug
        self.prune_broken_images()
    
    def prune_broken_images(self):
        clean_im_paths = []
        clean_lbls = []
        for im_pth, lbl in zip(self.im_paths, self.all_labels):
            full_img_pth = str(cfg.ROOT_CLF_IMG_DIR / im_pth)
            img = cv2.imread(full_img_pth)
            if img is not None:
                clean_im_paths.append(im_pth)
                clean_lbls.append(lbl)
        self.im_paths = clean_im_paths
        self.all_labels = clean_lbls

    def __getitem__(self, item_idx):
        img_pth = str(cfg.ROOT_CLF_IMG_DIR / self.im_paths[item_idx])
        img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.im_w, self.im_h))
        raw_img = np.copy(img)
        label = self.all_labels[item_idx]
        
        if self.debug:
            print(f"img.shape: {img.shape}")
            fig = plt.figure(figsize=(8,8))
            fig.add_subplot(2, 2, 1)
            plt.imshow(img.astype(np.uint8))
            plt.title("Original Image")
        
        # Preprocess the image according to imagenet mean/var
        if self.img_preproc_funcs is not None:
            for f in self.img_preproc_funcs:
                img = f(img)
        
        if self.debug:
            fig.add_subplot(2, 2, 2)
            plt.imshow((img * 255).astype(np.uint8))
            plt.title("Preprocessed Image")
            debug_dir = cfg.CLF_OUTPUT_DIR / "debug_plots"
            if not debug_dir.exists():
                debug_dir.mkdir(parents=True)
            fig.savefig(str(debug_dir / f'{self.dataset_type}_debug_img_mask_{item_idx}.png'))
        
        return img, label, raw_img

    def __len__(self):
        return len(self.im_paths)
    
    def __add__(self, other):
        if other.im_w != self.im_w or other.im_h != self.im_h:
            raise ValueError("Can not add datasets with differing desired image sizes")
        if other.img_preproc_funcs != self.img_preproc_funcs:
            raise ValueError("Cannot add datasets with different preprocessing functions")
        im_paths = self.im_paths + other.im_paths
        all_labels = self.all_labels + other.all_labels
        combo = RawDataset(im_paths, all_labels, image_size=(self.im_w, self.im_h), preprocessing=self.img_preproc_funcs)
        return combo


class RawDataLoader(keras.utils.Sequence):
    def __init__(self, dataset, preds, batch_size=32, save_masks=True):
        self.dataset = dataset
        self.preds = preds
        self.batch_size = batch_size if batch_size <= len(dataset) else len(dataset)
        if self.preds is not None:
            assert len(self.dataset) == len(self.preds), "The dataset and predictions must be of equal size"
        self.save_masks = save_masks

    def __getitem__(self, item_idx):
        start, stop = item_idx * self.batch_size, (item_idx + 1) * self.batch_size
        if stop > len(self.dataset):
            stop = len(self.dataset)

        batch_imgs, batch_raw_imgs, batch_img_paths, batch_pred_masks, batch_mask_paths, batch_labels = [], [], [], [], [], []
        for i in range(start, stop):
            img, label, raw_img = self.dataset[i]
            batch_imgs.append(img)
            batch_raw_imgs.append(raw_img)
            batch_img_paths.append(str(cfg.ROOT_CLF_IMG_DIR / self.dataset.im_paths[i]))
            batch_labels.append(label)

            if self.preds is not None:
                pred_mask = bounded_decision_function(self.preds[i], threshold=cfg.BOUNDED_DECISION_THRESHOLD)
                base_mask_path = self.dataset.im_paths[i].split("/")[-1]
                base_mask_path = base_mask_path.split(".")[:-1]
                base_mask_path.append(".npy")
                base_mask_path = "".join(base_mask_path)
                pred_mask_fname = str(cfg.CLF_MASK_DIR / self.dataset.dataset_type / base_mask_path)
                if self.save_masks:
                    np.save(pred_mask_fname, pred_mask)
                batch_mask_paths.append(pred_mask_fname)
                batch_pred_masks.append(pred_mask)
            
        if self.preds is None:
            return np.asarray(batch_imgs), np.asarray(batch_raw_imgs)
        else:
            return np.asarray(batch_raw_imgs), batch_img_paths, np.asarray(batch_pred_masks), batch_mask_paths, batch_labels

    def __len__(self):
        return (len(self.dataset) // self.batch_size) + (1 if len(self.dataset) % self.batch_size > 0 else 0)


def load_raw_splits():
    train_df = pd.read_csv(str(cfg.CLF_RAW_SPLITS_DIR / 'NO_vs_EM_vs_HZ_vs_TC_train.csv'))
    train_df.columns = ['images', 'disease_labels']
    val_df = pd.read_csv(str(cfg.CLF_RAW_SPLITS_DIR / 'NO_vs_EM_vs_HZ_vs_TC_val.csv'))
    val_df.columns = ['images', 'disease_labels']
    test_df = pd.read_csv(str(cfg.CLF_RAW_SPLITS_DIR / 'NO_vs_EM_vs_HZ_vs_TC_test.csv'))
    test_df.columns = ['images', 'disease_labels']

    train_set = RawDataset(
        im_paths=train_df['images'].to_list(), 
        labels=train_df['disease_labels'].to_list(),
        image_size=cfg.IMAGE_SIZE,
        img_preprocessing=[preprocess_input], # Imagenet normalization
        dataset_type='train',
        debug=False)
    
    val_set = RawDataset(
        im_paths=val_df['images'].to_list(), 
        labels=val_df['disease_labels'].to_list(), 
        image_size=cfg.IMAGE_SIZE,
        img_preprocessing=[preprocess_input],
        dataset_type='val')

    test_set = RawDataset(
        im_paths=test_df['images'].to_list(), 
        labels=test_df['disease_labels'].to_list(), 
        image_size=cfg.IMAGE_SIZE,
        img_preprocessing=[preprocess_input],
        dataset_type='test')
        
    return train_set, val_set, test_set

def gen_preliminary_data():
    # Calculate ITA and store masks as npy arrays
    datasets = {'train':[], 'val':[], 'test':[]}
    datasets['train'], datasets['val'], datasets['test'] = load_raw_splits()
    
    ita_bins_data = get_ita_data()  # loads in the data for each ITA category

    s = time.time()
    for split in ['train', 'val', 'test']:
        # Predicting all masks at once is too memory intensive, so we do it in batches
        raw_loader = RawDataLoader(datasets[split], None, batch_size=cfg.SKIN_SEG_BATCH_SIZE, save_masks=False)
        preds = predict_mask_generator(None, custom_loader=raw_loader, batch_size=cfg.SKIN_SEG_BATCH_SIZE, make_masks_usable=False)
        (cfg.CLF_MASK_DIR / split).mkdir(parents=True, exist_ok=True)
        loader = RawDataLoader(datasets[split], preds, batch_size=cfg.SKIN_SEG_BATCH_SIZE, save_masks=True)
        df = pd.DataFrame(columns=['image', 'pred_mask', 'disease_label', 'pred_ita', 'pred_skintone_category', 'pred_binary_skintone'])  # creates the dataframe

        for content in tqdm(loader, desc="Categorizing images"):
            batch_raw_imgs, batch_img_paths, batch_pred_masks, batch_mask_paths, batch_labels = content
            
            pred_ita_vals = calc_batch_ITAs(batch_raw_imgs, batch_pred_masks)

            # Saves the skin tone category for each image to the dataframe (categories defined in
            # skin_tone_categorization_scheme.txt, which is pulled from Table 1 in the original paper).
            for img_path, mask_path, lbl, pred_ita in zip(batch_img_paths, batch_mask_paths, batch_labels, pred_ita_vals):
                pred_tone = 'undef'
                for k in ita_bins_data.keys():
                    if ita_bins_data[k]['check_in_bin'](pred_ita):  # checks which bin the current ITA value falls under
                        pred_tone = k
                
                binary_skintone = 'ds' if pred_tone in ['tan2', 'tan1', 'dark'] else 'ls'

                df = df.append({
                    'image':str(img_path),
                    'pred_mask':str(mask_path),
                    'disease_label':lbl,
                    'pred_ita':pred_ita,
                    'pred_skintone_category':pred_tone,
                    'pred_binary_skintone':binary_skintone}, ignore_index=True) # Apparently df.append() doesn't append in place...y tho

        csv_name = str(cfg.CLF_SPLITS_DIR / f"{split}_processed.csv") #str(cfg.SKIN_TONE_CSV).replace('skin_tones', 'test_skin_tones')
        df.to_csv(csv_name, columns=df.columns, index=False) # save each iteration in case errors occur later

    print(f"Masks generated and ITA calculated in {time.time() - s} seconds")


# ////////////////////////////////////////////////
# Clean Data
# ////////////////////////////////////////////////

class SkinDiseaseDataset(T.utils.data.Dataset):
    def __init__(self, images, labels, pred_mask_paths, pred_binary_skintone, split="all", image_size=cfg.IMAGE_SIZE,
                 img_preprocessing=True, mask_images=True, debug=False, edgemixup = False):
        self.im_w, self.im_h = image_size
        self.split = split
        self.im_paths = images
        self.labels = labels
        self.pred_mask_paths = pred_mask_paths
        self.pred_binary_skintone = pred_binary_skintone
        self.mask_images = mask_images
        self.debug = debug
        if img_preprocessing:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.inv_transform = transforms.Compose([
                NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = None
            self.inv_transform = None
        self.edgemixup = edgemixup

    
    def __len__(self):
        return len(self.im_paths)
    
    def __getitem__(self, idx):
        img_pth = str(Path(cfg.ROOT_CLF_IMG_DIR) / self.im_paths[idx])
        img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.im_w, self.im_h))
        mask_path = self.pred_mask_paths[idx]# if ".npy" in self.pred_mask_paths[idx] else self.pred_mask_paths[idx] + ".npy"
        pred_mask = np.load(mask_path)
        assert (img.shape[0:2] == pred_mask.shape[0:2]), f"Image shape {img.shape} and mask shape {pred_mask.shape} differ"

        if self.debug:
            fig = plt.figure(figsize=(8,8))
            fig.add_subplot(2, 2, 1)
            plt.imshow(img.astype(np.uint8))
            plt.title(f"Original Image\n\t{ ''.join(self.im_paths[idx].split('/')[-2:]) }")
            fig.add_subplot(2, 2, 3) 
            plt.imshow((pred_mask!= cfg.Label.LESION.value).astype(np.uint8) * 127)
            plt.title(f"Original Mask\n\t{ ''.join(self.pred_mask_paths[idx].split('/')[-2:]) }\n")

        if self.split == "train":
            img, pred_mask = clf_im_aug_func(img, pred_mask)
        if self.transform is not None:
            img = self.transform(img).numpy()
        
        if self.debug:
            fig.add_subplot(2, 2, 2)
            plt_img = self.inv_transform(T.tensor(img)).numpy() if self.inv_transform is not None else img
            plt_img = plt_img * 255
            plt.imshow(np.transpose(plt_img, (1, 2, 0)).astype(np.uint8))
            plt.title(f"Processed Image")
        
        if self.mask_images:
            pred_mask = np.transpose(pred_mask, (2, 0, 1))
            mask = np.tile(pred_mask, (3, 1, 1)) != cfg.Label.LESION.value # Repeat the mask across all 3 RGB dimensions
            img = np.ma.masked_where(mask, img)
            img = img.filled(fill_value=0.0).astype("float32")

        #if self.transform is not None:
        #    img = self.transform(T.tensor(img)).numpy()

        label = np.asarray(self.labels[idx])
        pred_skintone = self.pred_binary_skintone[idx] == "ls" # 0 : "ds", 1: "ls"
        pred_skintone = np.asarray(pred_skintone).astype("int64")

        if self.debug:
            fig.add_subplot(2, 2, 4)
            plt_img = self.inv_transform(T.tensor(img)).numpy() if self.inv_transform is not None else img
            plt_img = plt_img * 255
            plt.imshow(np.transpose(plt_img, (1, 2, 0)).astype(np.uint8))
            plt.title(f"Masked Processed Image. Label: {cfg.CLF_Label_Translate[self.labels[idx]]} Skintone: {self.pred_binary_skintone[idx]} idx: {idx}")
            debug_dir = cfg.CLF_OUTPUT_DIR / "debug_plots" / self.split
            if not debug_dir.exists():
                debug_dir.mkdir(parents=True)
            fig.savefig(str(debug_dir / f'{self.split}_debug_img_mask_{idx}.png'))
            plt.cla()

        return img, label, pred_skintone

class EvalSkinDiseaseDataset():
    def __init__(self, images, labels, pred_mask_paths, pred_binary_skintone, pred_disease, image_size=cfg.IMAGE_SIZE, mask_images=True, model_type="baseline"):
        self.im_w, self.im_h = image_size
        self.im_paths = images
        self.labels = labels
        self.pred_mask_paths = pred_mask_paths
        self.pred_binary_skintone = pred_binary_skintone
        self.pred_disease = pred_disease
        self.mask_images = mask_images
        self.model_type = model_type
    
    def __len__(self):
        return len(self.im_paths)
    
    def __getitem__(self, idx):
        img_pth = str(cfg.ROOT_CLF_IMG_DIR / self.im_paths[idx])
        img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.im_w, self.im_h))
        mask_path = self.pred_mask_paths[idx]# if ".npy" in self.pred_mask_paths[idx] else self.pred_mask_paths[idx] + ".npy"
        pred_mask = np.load(mask_path)

        fig = plt.figure(figsize=(10,10))
        fig.add_subplot(2, 2, 1)
        plt.imshow(img.astype(np.uint8))
        raw_img = img.astype(np.uint8)
        #plt.title(f"Original Image\n\t{ ''.join(self.im_paths[idx].split('/')[-2:]) }")
        plt.title(f"Original Image")
        fig.add_subplot(2, 2, 3) 
        plt.imshow((pred_mask!= cfg.Label.LESION.value).astype(np.uint8) * 127)
        #plt.title(f"Original Mask\n\t{ ''.join(self.pred_mask_paths[idx].split('/')[-2:]) }\n")
        plt.title(f"Original Mask")

        if self.mask_images:
            img = np.transpose(img, (2, 0, 1))
            pred_mask = np.transpose(pred_mask, (2, 0, 1))
            mask = np.tile(pred_mask, (3, 1, 1)) != cfg.Label.LESION.value # Repeat the mask across all 3 RGB dimensions
            img = np.ma.masked_where(mask, img)
            img = img.filled(fill_value=0.0).astype("float32")
        else:
            img = np.transpose(img, (2, 0, 1)).astype("float32")

        label = np.asarray(self.labels[idx])
        pred_skintone = self.pred_binary_skintone[idx] == "ls" # 0 : "ds", 1: "ls"
        pred_skintone = np.asarray(pred_skintone).astype("int64")
        pred_disease_prob = T.tensor(self.pred_disease[idx, :])
        pred_disease_prob = softmax(pred_disease_prob, dim=-1)
        top_k_prob, top_k_idx = pred_disease_prob.topk(1, dim=0)
        pred_disease_class = cfg.CLF_Label_Translate[top_k_idx.item()]

        error_prediction = self.labels[idx] != top_k_idx.item()

        fig.add_subplot(2, 2, 4)
        plt.imshow(np.transpose(img, (1, 2, 0)).astype(np.uint8))
        #plt.title(f"\nPreprocessed Image. Label: {cfg.CLF_Label_Translate[self.labels[idx]]} Skintone: {self.pred_binary_skintone[idx]} idx: {idx}\n predicted disease: {pred_disease_class}")
        plt.title(f"\nPreprocessed Image. Label: {cfg.CLF_Label_Translate[self.labels[idx]]} Skintone: {self.pred_binary_skintone[idx]}\n predicted disease: {pred_disease_class}")
        debug_dir = cfg.CLF_OUTPUT_DIR / "test_result_images" / self.model_type if not error_prediction else cfg.CLF_OUTPUT_DIR / "test_result_images" / self.model_type / "error_classifying"
        debug_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(debug_dir / f'debug_test_img_mask_{idx}.png'))
        plt.cla()
        plt.close('all')

        result = {
            "img": raw_img,
            "masked_img": np.transpose(img, (1, 2, 0)),
            "pred": pred_disease_class,
            "gt": cfg.CLF_Label_Translate[self.labels[idx]],
            "skintone": self.pred_binary_skintone[idx],
            "correct": not error_prediction}

        return result

def load_clean_splits(model_type, edgemixup):
    if edgemixup:
        train_df = pd.read_csv(str(cfg.EdgeMixup_clf_dir / 'train_processed.csv'))
        val_df = pd.read_csv(str(cfg.EdgeMixup_clf_dir / 'val_processed.csv'))
        test_df = pd.read_csv(str(cfg.EdgeMixup_clf_dir / 'test_processed.csv'))
    else:
        train_df = pd.read_csv(str(cfg.CLF_SPLITS_DIR / 'train_processed.csv'))
        val_df = pd.read_csv(str(cfg.CLF_SPLITS_DIR / 'val_processed.csv'))
        test_df = pd.read_csv(str(cfg.CLF_SPLITS_DIR / 'test_processed.csv'))

    mask_images = "mask" in model_type
    datasets = {
        "train": SkinDiseaseDataset(
                    images = train_df['image'].to_list(),
                    labels = train_df['disease_label'].to_list(),
                    pred_mask_paths = train_df['pred_mask'].to_list(),
                    pred_binary_skintone = train_df['pred_binary_skintone'].to_list(),
                    image_size=cfg.IMAGE_SIZE,
                    img_preprocessing=[preprocess_input], # Imagenet normalization
                    split='train',
                    mask_images=mask_images,
                    debug=False),
        "val":  SkinDiseaseDataset(
                    images = val_df['image'].to_list(),
                    labels = val_df['disease_label'].to_list(),
                    pred_mask_paths = val_df['pred_mask'].to_list(),
                    pred_binary_skintone = val_df['pred_binary_skintone'].to_list(), 
                    image_size=cfg.IMAGE_SIZE,
                    img_preprocessing=[preprocess_input], # Imagenet normalization
                    split='val',
                    mask_images=mask_images),
        "test":  SkinDiseaseDataset(
                    images = test_df['image'].to_list(),
                    labels = test_df['disease_label'].to_list(),
                    pred_mask_paths = test_df['pred_mask'].to_list(),
                    pred_binary_skintone = test_df['pred_binary_skintone'].to_list(),
                    image_size=cfg.IMAGE_SIZE,
                    img_preprocessing=[preprocess_input], # Imagenet normalization
                    split='test',
                    mask_images=mask_images)
    }
        
    return datasets

def gen_eval_debug_plots(pred_disease, model_type):
    test_df = pd.read_csv(str(cfg.CLF_SPLITS_DIR / 'test_processed.csv'))

    mask_images = "mask" in model_type
    test_datasets = EvalSkinDiseaseDataset(
                        images = test_df['image'].to_list(),
                        labels = test_df['disease_label'].to_list(),
                        pred_mask_paths = test_df['pred_mask'].to_list(),
                        pred_binary_skintone = test_df['pred_binary_skintone'].to_list(),
                        pred_disease = pred_disease,
                        image_size = cfg.IMAGE_SIZE,
                        mask_images = mask_images,
                        model_type = model_type)
    
    debug_plot_dataloader = DataLoader(
        dataset=test_datasets,
        batch_size=16,
        num_workers=cpu_count(),
        pin_memory=False, # This dataset is small, no need
        drop_last=False,
        shuffle=False)
    
    fig = plt.figure(figsize=(10,10))
    num_incorrect_imgs = 1
    for i, result in tqdm(enumerate(debug_plot_dataloader)):
        for idx, correct in enumerate(result['correct']):
            if not correct:
                ax = fig.add_subplot(4, 4, num_incorrect_imgs)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(result['img'][idx].numpy().astype(np.uint8))
                plt.title(f"Pred: {result['pred'][idx]} GT: {result['gt'][idx]} ST: {result['skintone'][idx]}")
                num_incorrect_imgs += 1
                if "mask" in model_type:
                    ax = fig.add_subplot(4, 4, num_incorrect_imgs)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(result['masked_img'][idx].numpy().astype(np.uint8))
                    plt.title(f"Masked Image")
                    num_incorrect_imgs += 1
            if num_incorrect_imgs > 16:
                fig.tight_layout()
                debug_dir = cfg.CLF_OUTPUT_DIR / "test_result_images" / model_type / "error_classifying_montage"
                debug_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(debug_dir / f'test_img_montage_{i}.png'))
                num_incorrect_imgs = 1
                plt.cla()
                plt.clf()
                fig = plt.figure(figsize=(10,10))
    
    if num_incorrect_imgs > 1:
        fig.tight_layout()
        debug_dir = cfg.CLF_OUTPUT_DIR / "test_result_images" / model_type / "error_classifying_montage"
        debug_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(debug_dir / f'test_img_montage_{99}.png'))
        plt.cla()
    
    # Good examples
    fig = plt.figure(figsize=(10,10))
    num_incorrect_imgs = 1
    for i, result in tqdm(enumerate(debug_plot_dataloader)):
        for idx, correct in enumerate(result['correct']):
            if correct:
                ax = fig.add_subplot(4, 4, num_incorrect_imgs)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(result['img'][idx].numpy().astype(np.uint8))
                plt.title(f"Pred: {result['pred'][idx]} GT: {result['gt'][idx]}")
                num_incorrect_imgs += 1
                if "mask" in model_type:
                    ax = fig.add_subplot(4, 4, num_incorrect_imgs)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(result['masked_img'][idx].numpy().astype(np.uint8))
                    plt.title(f"Masked Image")
                    num_incorrect_imgs += 1
            if num_incorrect_imgs > 16:
                fig.tight_layout()
                debug_dir = cfg.CLF_OUTPUT_DIR / "test_result_images" / model_type / "correct_classifying_montage"
                debug_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(debug_dir / f'test_img_montage_{i}.png'))
                num_incorrect_imgs = 1
                plt.cla()
                plt.clf()
                fig = plt.figure(figsize=(10,10))
    
    if num_incorrect_imgs > 1:
        fig.tight_layout()
        debug_dir = cfg.CLF_OUTPUT_DIR / "test_result_images" / model_type / "correct_classifying_montage"
        debug_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(debug_dir / f'test_img_montage_{99}.png'))
        plt.cla()
    
    plt.close('all')
    
    print("Done generating eval test images.")