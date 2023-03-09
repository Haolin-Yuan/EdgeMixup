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
from skin_segmentation.model_utils import make_mask_usable,loss_function,JaccardScore,DICEScore
from skin_segmentation.data_utils import Dataset, DataLoader
from tensorflow.keras.applications.resnet50 import preprocess_input
import segmentation_models as sm
from tensorflow import keras
import skin_segmentation.segmentation_cfg as cfg
import pandas as pd
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import trange
from torch.nn.functional import softmax
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


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


def getHsvMask(path):
    alpha = cfg.alpah
    ori_img = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (256, 256))
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(img, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask1 = cv2.inRange(img, np.array([170, 50, 50]), np.array([180, 255, 255]))
    Redmask = mask0 + mask1
    img = cv2.Canny(Redmask, 100, 200)
    img = T.from_numpy(img).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
    img = (alpha * ori_img + (1 - alpha) * img).astype(np.uint8)
    return img


def generate_image_boundary_based_on_GT_label(df, folder_path):
    save_root = folder_path + "/image/"
    os.makedirs(save_root, exist_ok= True)
    alpha = 0.7
    save_df = df.copy()
    for i in tqdm(range(len(df.index))):
        img_path = df["images"][i]
        img = cv2.imread(img_path)
        img_name = Path(img_path).name
        mask = get_mask_from_json(img_path, df["target_masks"][i])
        gt_boundary = cv2.Canny(mask, 100, 200)
        gt_boundary = T.from_numpy(gt_boundary).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
        final_img = (alpha * img + (1 - alpha) * gt_boundary).astype(np.uint8)
        cv2.imwrite(save_root + img_name, final_img)
        save_df["images"][i] = save_root + img_name
    save_df.to_csv(folder_path + "/train.csv", index=False)



def generate_new_classification_sample(seg_model_path,file_root_path,img_save_path):
    '''
    :param seg_model_path:  the path of saved segmentation model
    :param file_root_path:  path of .csv files for all classification samples
    :param img_save_path:   path for saving new images
    :return:
    '''
    alpha = cfg.alpha
    model = keras.models.load_model(seg_model_path,
                                    custom_objects={'loss': loss_function(),
                                                    'jaccard_score': JaccardScore(),
                                                    f"DICE{int(cfg.METRIC_BETA)}_score": DICEScore(),
                                                    'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                    f"f{int(cfg.METRIC_BETA)}-score": sm.metrics.FScore(threshold=0.5)})
    index = 0
    for i in ["train"]:
        df = pd.read_csv(file_root_path+f"/{i}_processed.csv")
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
        for (data,_) in tqdm(test_loader):
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


seg_model_path = "/home/haolin/Projects/lyme/code/skin_segmentation/segmentation_runs/fitz17k_iter_1/best_model.h5"
file_root_path = "/home/haolin/Projects/lyme/code/fitzpatrick17k/clf_assets/processed_splits"
img_save_path = "/home/haolin/Projects/lyme/code/fitzpatrick17k/clf_assets/egded_iter_1_fitz17k/"



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

        # avg_RGB_threshold = (RGB_threshold / (j+1)).round(1)
        # avg_HSV_threshold = (HSV_threshold / (j+1)).round(1)
    avg_RGB_threshold = (RGB_threshold / len(df.index)).round(1)
    avg_HSV_threshold = (HSV_threshold / len(df.index)).round(1)

        # HSV_mask = cv2.inRange(HSV_img, avg_HSV_threshold[0], avg_HSV_threshold[1])
        # HSV_mask = T.from_numpy(HSV_mask).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
        # RGB_mask = cv2.inRange(RGB_img, avg_RGB_threshold[0], avg_RGB_threshold[1])
        # RGB_mask = T.from_numpy(RGB_mask).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
        # mask0 = cv2.inRange(HSV_img, np.array([0, 50, 50]), np.array([10, 255, 255]))
        # mask1 = cv2.inRange(HSV_img, np.array([170, 50, 50]), np.array([180, 255, 255]))
        # EdgeMixup = mask0 + mask1
        # EdgeMixup = T.from_numpy(EdgeMixup).unsqueeze(2).repeat(1, 1, 3).numpy().astype(np.uint8)
        # img = np.hstack((cv2.cvtColor(RGB_img, cv2.COLOR_RGB2BGR), EdgeMixup, cv2.bitwise_not(HSV_mask), cv2.bitwise_not(RGB_mask)))
        # cv2.imwrite(save_path + f"{img_name}.png", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
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
            # SD-198
            # transforms.Normalize((8.6501e-06,  1.3031e-04,  2.3714e-05), (1.0001, 0.9997, 1.0001))
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



def iterative_train_seg():

    pre_J = 0
    current_J = 0
    generate_new_classification_sample(seg_model_path,file_root_path,img_save_path)


def generate_edgemixup_class_data():
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



print("TESTGITHUB")