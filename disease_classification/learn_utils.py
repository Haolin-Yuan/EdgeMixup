# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import torch as T
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, Linear
from torch.nn.functional import softmax
from torchvision.models import resnet34
from multiprocessing import cpu_count
from tqdm import trange
import numpy as np

from disease_classification.data_utils import load_clean_splits, gen_eval_debug_plots
from disease_classification.models import AdvNet
from disease_classification.metrics import evaluate_performance_metrics
import disease_classification.clf_config as clss_config

import pdb
from edge_mixup_util import *

def requires_grad(model, flag=True):
    for p in model.parameters():
            p.requires_grad = flag

def pretrain_AdvNet(dataloader, device, model, adv_model, adv_criterion, adv_optimizer):
    pbar = trange(clss_config.CLF_NUM_EPOCHS)
    model.train()
    adv_model.train()
    requires_grad(model, False)
    requires_grad(adv_model, True)

    for epoch in pbar:
        total_adv_loss = 0.0
        adv_loss = 0.0
        num_samples = 0.0
        for content in dataloader:
            imgs, labels, skintone = content
            num_samples = labels.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)
            skintone = skintone.to(device)
            
            preds = model(imgs)
            preds.requires_grad = True
            adv_preds = adv_model(preds, labels)

            adv_loss = adv_criterion(adv_preds, skintone)
            total_adv_loss += adv_loss.cpu().detach()
            model.zero_grad()
            adv_model.zero_grad()

            adv_loss.backward()
            adv_optimizer.step()
        
        pbar.set_description(f"AdvNet pretrain loss: {total_adv_loss / num_samples}")
    
    requires_grad(model, True)
    requires_grad(adv_model, True)
    return adv_model, adv_optimizer

def validate_model(train_AD, dataloader, model, criterion_model, device, early_stop_cnt, lowest_val_loss, adv_model, adv_criterion):
    total_val_loss = 0.0
    total_adv_val_loss = 0.0
    num_samples = 0.0
    model.eval()
    if train_AD:
        adv_model.eval()
    with T.no_grad():
        for content in dataloader:
            imgs, labels, skintone = content
            num_samples += labels.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)
            skintone = skintone.to(device)
            
            preds = model(imgs)
            if train_AD:
                adv_preds = adv_model(preds, labels)
                adv_loss = adv_criterion(adv_preds, skintone) * clss_config.CLF_ADV_LOSS_IMPORTANCE
                total_adv_val_loss += adv_loss.cpu().detach()
            else:
                adv_loss = 0.0
            
            loss = criterion_model(preds, labels) - adv_loss
            total_val_loss += loss.cpu().detach()
    
    mean_loss = total_val_loss / num_samples
    mean_adv_loss = total_adv_val_loss / num_samples
    if mean_loss >= lowest_val_loss:
        early_stop_cnt += 1
    else:
        early_stop_cnt = 0
        lowest_val_loss = mean_loss
    return early_stop_cnt, mean_loss, mean_adv_loss, lowest_val_loss


def train_model(args):
    if args.aux_model:
        print("***********************Training the extra model*********************")
        train_loader,val_loader,_ = get_AUX_training_data()
        dataloader = {"train":train_loader, "val":val_loader}
    else:
        datasets = load_clean_splits(args.model_type, args.edgemixup)
        dataloader = {
            dataset_type: T.utils.data.DataLoader(
                dataset=datasets[dataset_type],
                batch_size= clss_config.CLF_BATCH_SIZE
            ) for dataset_type in ['train', 'val']
        }


    device = T.device(f"cuda:{args.gpu}" if T.cuda.is_available() else "cpu")
    print(f"Training with device {device}")

    model = resnet34(pretrained=True) # Default pretrained is imagenet
    model.fc = Linear(512, len(clss_config.CLF_Label_Translate.keys()))
    model = model.to(device)

    criterion_model = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=clss_config.CLF_LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor= clss_config.CLF_LR_STEP_FACTOR,
        patience= clss_config.CLF_LR_SCHEDULER_PATIENCE)
    
    train_AD = "AD" in args.model_type
    if train_AD:
        adv_model = AdvNet(
            num_target_classes=clss_config.NUM_DISEASE_CLASSES,
            num_protected_classes=clss_config.NUM_PROTECTED_CLASSES).to(device)
        adv_criterion = CrossEntropyLoss()
        adv_optimizer = Adam(adv_model.parameters(), lr=clss_config.CLF_LR)
        adv_scheduler = lr_scheduler.ReduceLROnPlateau(
            adv_optimizer,
            mode="min",
            factor=clss_config.CLF_LR_STEP_FACTOR,
            patience=clss_config.CLF_LR_SCHEDULER_PATIENCE)
        adv_model, adv_optimizer = pretrain_AdvNet(dataloader["train"], device, model, adv_model, adv_criterion, adv_optimizer)
    else:
        adv_model, adv_criterion, adv_optimizer, adv_scheduler = None, None, None, None

    best_weights = {}
    prev_val_loss = 9.e9
    lowest_val_loss = 9.e9
    early_stop_cnt = 0
    best_epoch = 0
    last_loss = 0
    pbar = trange(clss_config.CLF_NUM_EPOCHS)
    for epoch in pbar:
        total_loss = 0.0
        total_adv_loss = 0.0
        loss = 0.0
        num_samples = 0.0
        model.train()
        if train_AD:
            adv_model.train()

        for content in dataloader['train']:
            imgs, labels, skintone = content
            num_samples += labels.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)
            skintone = skintone.to(device)
            
            preds = model(imgs)
            if train_AD:
                adv_preds = adv_model(preds, labels)
                adv_loss = adv_criterion(adv_preds, skintone) * clss_config.CLF_ADV_LOSS_IMPORTANCE
            else:
                adv_loss = 0.0
            
            loss = criterion_model(preds, labels) - adv_loss
            total_loss += loss.cpu().detach()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if train_AD:
                preds = model(imgs)
                adv_preds = adv_model(preds, labels)
                adv_loss = adv_criterion(adv_preds, skintone) * clss_config.CLF_ADV_LOSS_IMPORTANCE
                total_adv_loss += adv_loss.cpu().detach()
                adv_model.zero_grad()
                adv_loss.backward()
                adv_optimizer.step()
        
        last_loss = loss.detach().cpu()
        
        early_stop_cnt, prev_val_loss, prev_adv_val_loss, lowest_val_loss = validate_model(
                                                                                train_AD,
                                                                                dataloader['val'], 
                                                                                model, 
                                                                                criterion_model, 
                                                                                device,
                                                                                early_stop_cnt,
                                                                                lowest_val_loss,
                                                                                adv_model,
                                                                                adv_criterion)

        pbar.set_description(f"Training loss: {(total_loss / num_samples):4} Adv Loss: {(total_adv_loss / num_samples):4} Validation loss: {lowest_val_loss:4}")
        if early_stop_cnt == 1:
            best_epoch = epoch
            best_weights = {
                'epoch': best_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': last_loss,
                'adv_model': adv_model.state_dict() if train_AD else None,
                'adv_optimizer': adv_optimizer.state_dict() if train_AD else None,
                'adv_scheduler': adv_scheduler.state_dict() if train_AD else None
            }
        scheduler.step(prev_val_loss)
        if train_AD:
            adv_scheduler.step(prev_adv_val_loss)

        if early_stop_cnt >= 15:
            print(f"Early Stopping {epoch}")
            break

    if early_stop_cnt == 0:
        best_epoch = clss_config.CLF_NUM_EPOCHS
        best_weights = {
                'epoch': best_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': last_loss,
                'adv_model': adv_model.state_dict() if train_AD else None,
                'adv_optimizer': adv_optimizer.state_dict() if train_AD else None,
                'adv_scheduler': adv_scheduler.state_dict() if train_AD else None
            }
    if args.aux_model:
        weights_dir = str(clss_config.AUX_MODEL_DIR/f'{args.model_type}_best_model.pt')
    else:
        weights_dir = str(clss_config.CLF_OUTPUT_DIR / f'{args.model_type}_best_model.pt')
    T.save(best_weights, weights_dir)
    print(f"{weights_dir}")

def eval_model(args):
    device = T.device(f"cuda:{args.gpu}" if T.cuda.is_available() else "cpu")
    if args.aux_model:
        print("***********************Test extra model*********************")
        _,_, test_loader = get_AUX_training_data()
        dataloader = {"test":test_loader}
        weights_dir = str(clss_config.AUX_MODEL_DIR / f'{args.model_type}_best_model.pt')
    else:
        datasets = load_clean_splits(args.model_type)
        dataloader = {
            dataset_type: DataLoader(
                dataset=datasets[dataset_type],
                batch_size=clss_config.CLF_BATCH_SIZE
            ) for dataset_type in ['test']
        }
        weights_dir = str(clss_config.CLF_OUTPUT_DIR / f'{args.model_type}_best_model.pt')
    weights = T.load(weights_dir, map_location=device)
    model = resnet34(pretrained=True) # Default pretrained is imagenet
    model.fc = Linear(512, len(clss_config.CLF_Label_Translate.keys()))
    model.load_state_dict(weights['model'])
    model = model.to(device)
    criterion_model = CrossEntropyLoss()

    print(f"Loading weights: {weights_dir}")
    
    total_loss = 0.0
    num_samples = 0.0
    correct = 0.0

    all_preds = []
    all_lbls = []
    all_prot_factor = []
    model.eval()
    with T.no_grad():
        for content in dataloader['test']:
            imgs, labels, skintone = content
            num_samples += labels.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            preds = model(imgs)
            model.zero_grad()

            loss = criterion_model(preds, labels)
            total_loss += loss.cpu().detach()

            max_preds = softmax(preds, dim=-1).argmax(1).unsqueeze(-1)
            correct += max_preds.eq(labels.view_as(max_preds)).sum().cpu()
            all_preds.extend(preds.cpu().detach().tolist())
            all_lbls.extend(labels.cpu().detach().tolist())
            all_prot_factor.extend(skintone.detach().tolist())
    
    mean_loss = total_loss / num_samples
    accuracy = correct / num_samples
    print(f"Test loss: {mean_loss} accuracy: {accuracy}")

    if not args.aux_model:
        npy_all_preds = np.array(all_preds)
        npy_all_prot_factor = np.array(all_prot_factor)
        npy_all_lbls = np.array(all_lbls)

        gen_eval_debug_plots(npy_all_preds, args.model_type)

        results_dict, _ = evaluate_performance_metrics(npy_all_lbls, npy_all_preds, do_print=False)
        subset_indices_dict = {}
        subset_indices_dict['p0'] = np.where(npy_all_prot_factor == 0)[0]
        subset_indices_dict['p1'] = np.where(npy_all_prot_factor == 1)[0]
        subset_indices_dict['p0_t0'] = np.where(np.logical_and(npy_all_prot_factor == 0, npy_all_lbls == 0))[0]
        subset_indices_dict['p0_t1'] = np.where(np.logical_and(npy_all_prot_factor == 0, npy_all_lbls == 1))[0]
        subset_indices_dict['p1_t0'] = np.where(np.logical_and(npy_all_prot_factor == 1, npy_all_lbls == 0))[0]
        subset_indices_dict['p1_t1'] = np.where(np.logical_and(npy_all_prot_factor == 1, npy_all_lbls == 1))[0]

        results_sf0, _ = evaluate_performance_metrics(npy_all_lbls[subset_indices_dict['p0']], npy_all_preds[subset_indices_dict['p0']], do_print=False)
        results_sf1, _ = evaluate_performance_metrics(npy_all_lbls[subset_indices_dict['p1']], npy_all_preds[subset_indices_dict['p1']], do_print=False)

        # Baseline Metrics: TODO: updated this with unmasked
        '''
        baseline_acc = 85.10 (4.02)
        baseline_acc_gap = 13.15 (12.98)
        baseline_auc = 0.9725 (0.0185)
        baseline_auc_gap = 0.0331 (0.0714)
        '''

        baseline_acc = 85.10
        baseline_acc_gap = 13.15
        baseline_auc = 0.9725
        baseline_auc_gap = 0.0331

        print("==================================================")
        print(f" Model Type: {args.model_type}")
        print("==================================================")
        # Compute Metrics
        metric = 'Accuracy'
        print(f"{metric}\t{results_dict[metric][0][0]:.2f} ({results_dict[metric][0][1]:.2f})")
        gap = abs(results_sf0[metric][0][0] - results_sf1[metric][0][0])
        gap_error = abs(results_sf0[metric][0][1] - results_sf1[metric][0][1])
        print(f"{metric} Gap \t{gap:.2f} ({gap_error:.2f})")
        metric_min = 0
        metric_min_class = ""
        if results_sf0[metric][0][0] < results_sf1[metric][0][0]:
                metric_min = results_sf0[metric][0][0]
                metric_min_class = "ds"
        else:
                metric_min = results_sf1[metric][0][0]
                metric_min_class = "ls"
        print(f"{metric} min \t{metric_min:.2f} ({metric_min_class})")
        # CAI
        metric_alpha = 0.5
        cai = metric_alpha * (baseline_acc_gap - gap) + (1.0 - metric_alpha) * (results_dict[metric][0][0] - baseline_acc)
        print(f"CAI [{metric_alpha}]\t{cai:.4f}")
        metric_alpha = 0.75
        cai = metric_alpha * (baseline_acc_gap - gap) + (1.0 - metric_alpha) * (results_dict[metric][0][0] - baseline_acc)
        print(f"CAI [{metric_alpha}]\t{cai:.4f}")

        metric = 'AUC'
        print(f"{metric}\t\t{results_dict[metric][0][0]:.4f} ({results_dict[metric][0][1]:.4f})")
        gap = abs(results_sf0[metric][0][0] - results_sf1[metric][0][0])
        gap_error = abs(results_sf0[metric][0][1] - results_sf1[metric][0][1])
        print(f"{metric} Gap \t{gap:.4f} ({gap_error:.4f})")
        metric_min = 0
        metric_min_class = ""
        if results_sf0[metric][0][0] < results_sf1[metric][0][0]:
                metric_min = results_sf0[metric][0][0]
                metric_min_class = "ds"
        else:
                metric_min = results_sf1[metric][0][0]
                metric_min_class = "ls"
        print(f"{metric} min \t{metric_min:.4f} ({metric_min_class})")
        # CAUCI
        metric_alpha = 0.5
        cauci = metric_alpha * (baseline_auc_gap - gap) + (1.0 - metric_alpha) * (results_dict[metric][0][0] - baseline_auc)
        print(f"CAUCI [{metric_alpha}]\t{cauci:4f}")
        metric_alpha = 0.75
        cauci = metric_alpha * (baseline_auc_gap - gap) + (1.0 - metric_alpha) * (results_dict[metric][0][0] - baseline_auc)
        print(f"CAUCI [{metric_alpha}]\t{cauci:4f}")
