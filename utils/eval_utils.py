import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import EarlyStopping, Accuracy_Logger
from utils.file_utils import save_pkl, load_pkl
from sklearn.metrics import roc_auc_score, roc_curve, auc
import h5py
import math
from sklearn.preprocessing import label_binarize
# from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, accuracy_score, roc_auc_score

from models.model_AttnMIL import AttnMIL_Attention, AttnMIL_GatedAttention
from models.model_DeepAttnMISL import DeepAttnMIL
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_TransMIL import TransMIL
from models.model_DTFDMIL import DTFDMIL
from models.model_Surformer import MIL_Attention_FC_Surformer
from models.model_G_HANet import G_HANet
from models.model import *


def initiate_model(args, ckpt_path=None):
    print('Init Model')
    if args.model == 'AttnMIL':
        model = AttnMIL_Attention().to(device)
    elif args.model == 'AttnMIL_Gate':
        model = AttnMIL_GatedAttention().to(device)
    elif args.model == 'DeepAttnMIL':
        model = DeepAttnMIL().to(device)
    elif args.model == 'clam_sb':
        model_dict = {'n_classes': args.n_classes,
                      "embed_dim": args.embed_dim,
                      "size_arg": args.model_size}
        model = CLAM_SB(**model_dict).to(device)
    elif args.model == 'clam_mb':
        model_dict = {'n_classes': args.n_classes,
                      "embed_dim": args.embed_dim,
                      "size_arg": args.model_size}
        model_dict.update({"size_arg": args.model_size})
        model = CLAM_MB(**model_dict).to(device)
    elif args.model == 'TransMIL':
        model = TransMIL(n_classes=args.n_classes).to(device)
    elif args.model == 'DTFDMIL':
        model = DTFDMIL(n_classes=args.n_classes).to(device)
    elif args.model == 'Surformer':
        model = MIL_Attention_FC_Surformer(n_classes=args.n_classes).to(device)
    elif args.model == 'G_HANet':
        model_dict = {'n_classes': args.n_classes, 'num_tokens': args.num_tokens, 'ratio': args.ratio}
        model = G_HANet(**model_dict).to(device)
    elif args.model == 'LRENet':
        model_dict = {'n_classes': args.n_classes, 'num_tokens': args.num_tokens, 'ratio': args.ratio}
        model = LRENet(**model_dict).to(device)
    elif args.model == 'LRENet_adv':
        model_dict = {'n_classes': args.n_classes, 'num_tokens': args.num_tokens, 'ratio': args.ratio, 'wsi_size': args.embed_dim, 'agentDimList': args.agentDimList, 'agentList': args.agentList}
        model = LRENet_adv(**model_dict).to(device)
    print_network(model)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)

    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, ev_test_error, ev_auc_score, ev_precision, ev_recall, ev_f1, ev_mcc, ev_kappa, aucs, df, acc_logger= summary(model, loader, args)
    print('test_error: ', ev_test_error)
    print('auc: ', ev_auc_score)
    for cls_idx in range(len(aucs)):
        print('class {} auc: {}'.format(cls_idx, aucs[cls_idx]))
    return model, patient_results, ev_test_error, ev_auc_score, ev_precision, ev_recall, ev_f1, ev_mcc, ev_kappa, aucs, df


# Code taken from pytorch/examples for evaluating topk classification on on ImageNet
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    ev_test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, data in enumerate(loader):
        slide_id = slide_ids.iloc[batch_idx]
        share_feature = data[0].to(device, non_blocking=True)
        # source_feature = data[1].to(device, non_blocking=True)
        label = data[1].type(torch.LongTensor).to(device)
        with torch.no_grad():
            if args.model == 'LRENet':
                logits, Y_prob, Y_hat, moe_aux_loss, intersection_I, intersection_S, c_I, c_S = model(share_feature,
                                                                                            share_feature)
            elif args.model == 'LRENet_adv':
                logits, Y_prob, Y_hat, moe_aux_loss, kqv, kqv_c = model(
                    share_feature)
            else:
                logits, Y_prob, Y_hat = model(share_feature)

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        error = calculate_error(Y_hat, label)
        ev_test_error += error

    del data
    ev_test_error /= len(loader)

    ev_precision = 0.0
    ev_recall = 0.0
    ev_f1 = 0.0
    ev_mcc = 0.0
    ev_kappa = 0.0
    if args.n_classes == 2:
        # ev_test_error,ev_auc_score,ev_precision,ev_recall,ev_f1,ev_mcc,ev_kappa
        ev_precision = precision_score(all_labels, all_preds)
        ev_recall = recall_score(all_labels, all_preds)
        ev_f1 = f1_score(all_labels, all_preds)
        ev_mcc = matthews_corrcoef(all_labels, all_preds)
        ev_kappa = cohen_kappa_score(all_labels, all_preds)
    else:
        ev_precision = precision_score(all_labels, all_preds, average='macro')  # 宏平均精度
        ev_recall = recall_score(all_labels, all_preds, average='macro')  # 宏平均召回率
        ev_f1 = f1_score(all_labels, all_preds, average='macro')  # 宏平均F1分数
        ev_mcc = matthews_corrcoef(all_labels, all_preds)  # 马修斯相关系数
        ev_kappa = cohen_kappa_score(all_labels, all_preds)

        # if args.n_classes > 2:
    #     acc1, acc3 = accuracy(torch.from_numpy(all_probs), torch.from_numpy(all_labels), topk=(1, 3))
    #     print('top1 acc: {:.3f}, top3 acc: {:.3f}'.format(acc1.item(), acc3.item()))

    if len(np.unique(all_labels)) == 1:
        ev_auc_score = -1
    else:
        if args.n_classes == 2:
            ev_auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                ev_auc_score = auc(fpr, tpr)
            else:
                ev_auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return patient_results, ev_test_error, ev_auc_score, ev_precision, ev_recall, ev_f1, ev_mcc, ev_kappa, aucs, df, acc_logger
