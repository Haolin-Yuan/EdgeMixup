import config as cfg
import cv2
import numpy as np
import torch as T
from pathlib import Path
import os
import json
import zlib
import base64
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input
import segmentation_models as sm
from tensorflow import keras
import skin_segmentation.segmentation_cfg as cfg
import pandas as pd
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from torch.nn import Linear
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skin_segmentation.metrics_utils import JaccardScore, DICEScore
from skin_segmentation.losses import categorical_focal_loss

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


def make_mask_usable(model_pred):
    mask = np.argmax(model_pred, axis=-1)
    if mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def loss_function(num_classes=3):
    def loss(y_true, y_pred):
        dice_loss = sm.losses.DiceLoss(class_weights=cfg.DICE_LOSS_WEIGHTS)(y_true,
                                                                            y_pred)  # TODO: play with these coefficients be careful with these coefficients
        # focal_loss = sm.losses.BinaryFocalLoss()(y_true, y_pred) if num_classes == 1 else sm.losses.CategoricalFocalLoss()(y_true, y_pred) #use different implementation with alpha = tensor of size 3, simialr weighting to dice
        focal_loss = sm.losses.BinaryFocalLoss()(y_true, y_pred) if num_classes == 1 else categorical_focal_loss(
            alpha=cfg.FOCAL_LOSS_WEIGHTS)(y_true,
                                          y_pred)  # use different implementation with alpha = tensor of size 3, simialr weighting to dice
        total_loss = dice_loss + (cfg.SEG_LOSS_BETA * focal_loss)  # TODO: play with the beta

        return total_loss

    return loss

class Dataset:
    def __init__(self, im_paths, mask_paths=[], mask_json_dirs=cfg.MASK_JSON_DIRS, image_size=(600, 450),
                 augmentations=None, img_preprocessing=None, dataset_type="all", debug=False,
                 return_raw_imgs=False, edgeMixup=False):
        self.im_paths = im_paths
        if len(mask_paths) > 0:
            self.mask_paths = mask_paths
        else:
            self.mask_paths = []
            if mask_json_dirs is not None:
                for im_path in im_paths:
                    self.mask_paths.append(
                        Path(str(im_path.parent).replace('img', 'ann')) / (im_path.stem + '.png.json'))

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

            fig = plt.figure(figsize=(8, 8))
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

        combo = Dataset(im_paths, None, image_size=(self.im_w, self.im_h), augmentations=self.aug_funcs,
                        preprocessing=self.preproc_func)
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
            mask[msk_origin_y:msk_origin_y + curr_mask.shape[0], msk_origin_x:msk_origin_x + curr_mask.shape[1]][
                curr_mask] = fill_val.value

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

def base64_2_mask(s): # from https://docs.supervise.ly/data-organization/import-export/supervisely-format
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, dtype=np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def get_mask_from_json(img_path, mask_json_path):
    with open(str(mask_json_path)) as f:
        json_dict = json.load(f)
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2])
    for mask_dict in json_dict['objects']:
        if mask_dict['classTitle'] == "Lesion":
            bmp_str = mask_dict['bitmap']['data']
            curr_mask = base64_2_mask(bmp_str)
            msk_origin_x, msk_origin_y = mask_dict['bitmap']['origin']
            mask_label = mask_dict['classTitle']
            fill_val = cfg.Label.LESION if mask_label == 'Lesion' else cfg.Label.SKIN
            mask[msk_origin_y:msk_origin_y + curr_mask.shape[0], msk_origin_x:msk_origin_x + curr_mask.shape[1]][curr_mask] = fill_val.value
    mask = T.from_numpy(mask).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8) * 255
    return mask


def getHsvMask(img):
    alpha = cfg.alpha
    ori_img = img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask0 = cv2.inRange(img, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask1 = cv2.inRange(img, np.array([170, 50, 50]), np.array([180, 255, 255]))
    Redmask = mask0 + mask1
    img = cv2.Canny(Redmask, 100, 200)
    img = T.from_numpy(img).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
    img = (alpha * ori_img + (1 - alpha) * img).astype(np.uint8)
    return img


def predict_mask(img, model_path):
    model = keras.models.load_model(str(cfg.MODEL_PATH),
                                    custom_objects={'loss': loss_function(),
                                                    'jaccard_score': JaccardScore(),
                                                    f"DICE{int(cfg.METRIC_BETA)}_score": DICEScore(),
                                                    'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                    f"f{int(cfg.METRIC_BETA)}-score": sm.metrics.FScore(threshold=0.5)})
    model_input = img
    model.run_eagerly = True

    if len(model_input.shape) == 4:
        pred = model.predict(model_input)
    else:
        pred = model.predict(np.expand_dims(model_input, axis=0))

    return model_input, pred


def add_segment_boundary(df, save_root, test_model_path):
    os.makedirs(os.path.dirname(save_root), exist_ok=True)
    alpha = 0.7
    save_df = df.copy()
    for i in tqdm(range(len(df.index))):
        img_path = df["images"][i]
        img_name = Path(img_path).name
        img_name = ".".join(img_name.split(".")[::len(img_name.split("."))-1])
        save_df["images"][i] = save_root +img_name
        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # cv2.resize reverse width and length, thus getting original size by [1::-1]
        original_size = img.shape[1::-1]
        img = cv2.resize(img, (256, 256))
        pure_img = img.copy()
        _, model_pred = predict_mask(img, model_path=test_model_path)
        pr_mask = make_mask_usable(model_pred)
        pr_mask = np.where(pr_mask == 1, 255, 0)
        pr_mask = cv2.Canny(np.uint8(pr_mask), 100, 200)
        pr_mask = T.from_numpy(pr_mask).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
        final_img = (alpha * pure_img + (1 - alpha) * pr_mask).astype(np.uint8)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        final_img = cv2.resize(final_img, original_size)
        cv2.imwrite(save_root + img_name, final_img)

    return save_df



def generate_new_classification_sample(seg_model_path,file_root_path,img_save_path):
    '''
    :param seg_model_path:  the path of saved segmentation model
    :param file_root_path:  path of .csv files for all classification samples
    :param img_save_path:   path for saving new images
    :return: new df file containing new training sample directory
    '''
    alpha = cfg.alpha
    model = keras.models.load_model(seg_model_path,
                                    custom_objects={'loss': loss_function(),
                                                    'jaccard_score': JaccardScore(),
                                                    f"DICE{int(cfg.METRIC_BETA)}_score": DICEScore(),
                                                    'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                    f"f{int(cfg.METRIC_BETA)}-score": sm.metrics.FScore(threshold=0.5)})
    index = 0

    df = pd.read_csv(file_root_path+f"/train_processed.csv")
    df_2 = df.copy()
    data_set = Dataset(
        im_paths=df['image'].to_list(),
        mask_paths=df['pred_mask'].to_list(),
        image_size= (256,256),
        img_preprocessing=[preprocess_input],  # Imagenet normalization
        augmentations=[],
        dataset_type='test',
        debug=False,
        return_raw_imgs=False,
        edgeMixup=False)
    test_loader = DataLoader(data_set, batch_size=8, shuffle_every_epoch=False)
    for data,_ in tqdm(test_loader):
        model_pred = model.predict(data)
        model_pred = np.argmax(model_pred, axis=-1)
        for j in range(data.shape[0]):
            pr_mask = model_pred[j]
            pr_mask = np.where(pr_mask == 1, 255, 0)
            pr_mask = cv2.Canny(np.uint8(pr_mask), 100, 200)
            pr_mask = T.from_numpy(pr_mask).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
            final_img = (alpha * data[j] + (1 - alpha) * pr_mask).astype(np.uint8)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            df_2["image"][index] = img_save_path + f"{id}.png"
            cv2.imwrite(img_save_path, final_img)
            index += 1
        return df_2



def general_lesion_color(df):
    # if not os.path.exists(save_path): os.makedirs(save_path)
    HSV_threshold = np.zeros((2, 3))
    RGB_threshold = np.zeros((2, 3))

    for j in tqdm(range(len(df.index))):
        img_path = df["images"][j]
        img = cv2.imread(img_path)
        mask_path = df["target_masks"][j]
        mask = get_mask_from_json(img_path, mask_path)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # masked_img = cv2.bitwise_and(RGB_img, mask)
        for i in range(3):
            HSV_threshold[0][i] += HSV_img[:, :, i][mask[:,:,0]== 255].min()
            HSV_threshold[1][i] += HSV_img[:, :, i][mask[:,:,0]== 255].max()
            RGB_threshold[0][i] += RGB_img[:, :, i][mask[:,:,0]== 255].min()
            RGB_threshold[1][i] += RGB_img[:, :, i][mask[:,:,0]== 255].max()

    avg_RGB_threshold = (RGB_threshold / len(df.index)).round(1)
    avg_HSV_threshold = (HSV_threshold / len(df.index)).round(1)

    return avg_RGB_threshold, avg_HSV_threshold


def get_mean_stats_from_seg(segm_folder_path):
    """It gets the color threhsold from both HSV and RGB image"""
    df = pd.DataFrame()
    for i in os.listdir(segm_folder_path):
        df_temp = pd.read_csv(segm_folder_path + "/" + i).copy()
        df = pd.concat([df, df_temp], ignore_index=True)
    avg_RGB_threshold, avg_HSV_threshold = general_lesion_color(df)
    return avg_RGB_threshold, avg_HSV_threshold


def mean_color_mask(path, HSV_threshold, RGB_threshold):
    ori_img = cv2.resize(cv2.imread(path), (256, 256))
    HSV_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(HSV_img, HSV_threshold[0], HSV_threshold[1])

    RGB_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    RGB_mask = cv2.inRange(RGB_img, RGB_threshold[0], RGB_threshold[1])
    img_list = []
    for mask in [HSV_mask, RGB_mask]:
        boundary = cv2.Canny(mask, 100, 200)
        boundary = T.from_numpy(boundary).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
        img = (cfg.alpha * ori_img + (1 - cfg.alpha) * boundary).astype(np.uint8)
        img_list.append(img)
    return img_list[0], img_list[1]


class Aux_Dataset(T.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.images = self.df["images"]
        self.targets = self.df["label"]
        self.mask = self.df["target_masks"]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, item):
        img_fn = self.images[item]
        label = self.targets[item]
        mask = self.mask[item]
        mask = get_mask_from_json(img_fn, mask)
        mask = cv2.Canny(mask, 100, 200)
        mask = T.from_numpy(mask).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
        img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
        img = (0.7 * img + 0.3 * mask).astype(np.uint8)
        img = cv2.resize(img, (255, 255))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def get_AUX_training_data():
    list = []
    root = cfg.ROOT_DATA_DIR
    label_dict = {}
    cfg.AUX_DATA_DIR
    for i, name in enumerate(os.listdir(root)): label_dict[f'{name}']= i
    for i in ["train", "val", "test"]:
        df = pd.read_csv(str(root/i)+".csv").copy()
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.drop(["gt_binary_skintone"], axis = 1)
        df.insert(loc = 1, column = "label", value = 0)
        for row in range(len(df.index)):
            df["label"][row] = label_dict[df["images"][row].split('/')[-3]]
        df.to_csv(f"/home/haolin/Projects/lyme/code/extra_model/SD-198_reduced_5/{i}_set.csv")
        list.append(df)

    train_loader = DataLoader(Aux_Dataset(list[0]), batch_size=8, num_workers=1, pin_memory=False)
    val_loader = DataLoader(Aux_Dataset(list[1]), batch_size=8, num_workers=1, pin_memory=False)
    test_loader = DataLoader(Aux_Dataset(list[2]), batch_size=8, num_workers=1, pin_memory=False)
    return train_loader, val_loader, test_loader


def getting_optimal_edge():
    '''
    Choosing an optimal lesion edge as for the first-iteration segmentation training
    :return:
    '''
    weights = T.load(cfg.AUX_MODEL_DIR, map_location=device)
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = Linear(512, 4)
    model = model.to(device)
    model.load_state_dict(weights['model'])
    model.eval()

    avg_RGB_threshold, avg_HSV_threshold = get_mean_stats_from_seg()
    for file in os.listdir(cfg.DATA_ASSETS_DIR):
        df = pd.read_csv(str(cfg.DATA_ASSETS_DIR/file)).copy()
        for i in range(len(df.index)):
            # generate mask candidatas
            img_pth = df["images"][i]
            img_name = Path(img_pth).name
            HSV_img, RGB_img = mean_color_mask(img_pth, avg_HSV_threshold, avg_RGB_threshold)
            EdgeMixup_img = getHsvMask(img_pth)
            # choose best mask candidiate
            mask_list = []
            transform = transforms.Compose([transforms.ToTensor()])
            with T.no_grad():
                for img in [EdgeMixup_img, HSV_img, RGB_img]:
                    img = T.unsqueeze(transform(img), 0).to(device)
                    pred = model(img)
                    mask_list.append(T.max(pred))
            index = mask_list.index(max(mask_list))
            if index == 0:
                img = EdgeMixup_img
            elif index == 1:
                img = HSV_img
            else:
                img = RGB_img
            cv2.imwrite(cfg.EdgeMixup_clf_data +"/" +img_name, img)
            df["images"][i] = cfg.EdgeMixup_clf_data + "/" + img_name
        df.to_csv(str(cfg.EdgeMixup_clf_dir/file))


def clean_df(df):
    for i in range(len(df.index)):
        if df["images"][i].split(".")[-1] not in ["jpg", "png", "jpeg"]:
            df = df.drop([i])
    df = df.reset_index(drop=True)
    return df


