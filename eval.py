from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataProcess.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
# parser.add_argument('--data_root_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/patch_feature/',
#                     help='data directory')
# parser.add_argument('--data_slide_dir', type=str,
#                     default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/svs/')
# parser.add_argument('--results_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/result/task_gbmlgg/',
#                     help='relative path to results folder, i.e. '+
#                     'the directory containing models_exp_code relative to project root (default: ./results)')
#
# parser.add_argument('--save_exp_code', type=str, default='LRENet_task_gbmlgg_s1',
#                     help='experiment code to save eval results')
# parser.add_argument('--models_exp_code', type=str, default='LRENet_task_gbmlgg_s1',
#                     help='experiment code to load trained models (directory under results_dir containing model checkpoints')
#
# parser.add_argument('--splits_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/splits/task_gbmlgg_100/',
#                     help='splits directory, if using custom splits other than what matches the task (default: None)')
# parser.add_argument('--task', type=str, default='task_gbmlgg', choices=['task_gbmlgg', 'task_pcaepe'])


parser.add_argument('--data_root_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_conch/',
                    help='data directory')
parser.add_argument('--data_slide_dir', type=str, default='/data1/xiamy/PCa-EPE/WSI/')
parser.add_argument('--results_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/result/task_pcaepe/',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')

parser.add_argument('--save_exp_code', type=str, default='LRENet_adv_task_pcaepe_e91_s1_conch',
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default='LRENet_adv_task_pcaepe_e91_s1',
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')

parser.add_argument('--splits_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/splits/task_pcaepe_100/',
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--task', type=str, default='task_pcaepe', choices=['task_gbmlgg', 'task_pcaepe'])


parser.add_argument('--model', type=str, default='LRENet_adv',
                    choices=['AttnMIL', 'AttnMIL_Gate', 'DeepAttnMIL', 'clam_sb', 'clam_mb', 'TransMIL', 'DTFDMIL', 'Surformer', 'G_HANet', 'LRENet', 'LRENet_adv'])

# choices=['resnet50_trunc', 'uni_v1', 'uni2-h', 'conch_v1', 'conch_v1_5', 'CHIEF-Ctranspath', 'MUSK']
parser.add_argument('--MultiAgent', default=True)
# parser.add_argument('--Agent1', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_uni')
# parser.add_argument('--Agent2', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_Virchow')
parser.add_argument('--Agent3', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_Virchow2')
parser.add_argument('--Agent4', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_prov-gigapath')
parser.add_argument('--Agent5', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_uni2')
parser.add_argument('--Agent6', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_conch/')
parser.add_argument('--Agent7', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_TITAN')
parser.add_argument('--Agent8', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_CHIEF/')
parser.add_argument('--Agent1', type=str, default=None)
parser.add_argument('--Agent2', type=str, default=None)
parser.add_argument('--Agent1_dim', type=int, default=0)
parser.add_argument('--Agent2_dim', type=int, default=0)
parser.add_argument('--Agent3_dim', type=int, default=2560)
parser.add_argument('--Agent4_dim', type=int, default=1536)
parser.add_argument('--Agent5_dim', type=int, default=1536)
parser.add_argument('--Agent6_dim', type=int, default=512)
parser.add_argument('--Agent7_dim', type=int, default=768)
parser.add_argument('--Agent8_dim', type=int, default=768)

parser.add_argument('--num_tokens', type=int, default=6, help='Number of tokens ')
parser.add_argument('--ratio', type=float, default=0.2, help='For margin loss ')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--embed_dim', type=int, default=512)
args = parser.parse_args()

device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join(args.results_dir + '/eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

agentList = list([])
agentDimList = list([])
if args.MultiAgent is True:
    if args.Agent1_dim > 0:
        agentList.append(args.Agent1)
        agentDimList.append(args.Agent1_dim)
    if args.Agent2_dim > 0:
        agentList.append(args.Agent2)
        agentDimList.append(args.Agent2_dim)
    if args.Agent3_dim > 0:
        agentList.append(args.Agent3)
        agentDimList.append(args.Agent3_dim)
    if args.Agent4_dim > 0:
        agentList.append(args.Agent4)
        agentDimList.append(args.Agent4_dim)
    if args.Agent5_dim > 0:
        agentList.append(args.Agent5)
        agentDimList.append(args.Agent5_dim)
    if args.Agent6_dim > 0:
        agentList.append(args.Agent6)
        agentDimList.append(args.Agent6_dim)
    if args.Agent7_dim > 0:
        agentList.append(args.Agent7)
        agentDimList.append(args.Agent7_dim)
    if args.Agent8_dim > 0:
        agentList.append(args.Agent8)
        agentDimList.append(args.Agent8_dim)
    args.agentList = agentList
    args.agentDimList = agentDimList

print(settings)
if args.task == 'task_gbmlgg':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/LGG-GBM.csv',
                                  data_dir=os.path.join(args.data_root_dir, ''),
                                  data_slide_dir=os.path.join(args.data_slide_dir, ''),
                                  target_patch_size=args.target_patch_size,
                                  use_h5=False,
                                  MultiAgent=args.MultiAgent,
                                  agentList=agentList,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'Dead': 0, 'Alive': 1},
                                  patient_strat=False,
                                  ignore=[])
elif args.task == 'task_pcaepe':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/PCa-EPE_use.csv',
        data_dir=os.path.join(args.data_root_dir, ''),
        data_slide_dir=os.path.join(args.data_slide_dir, ''),
        target_patch_size=args.target_patch_size,
        use_h5=False,
        MultiAgent=args.MultiAgent,
        agentList=agentList,
        shuffle=False,
        print_info=True,
        label_dict={'C1': 0, 'C2': 1},
        patient_strat=False,
        ignore=[])
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_mcc = []
    all_kappa = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, ev_test_error, ev_auc_score, ev_precision, ev_recall, ev_f1, ev_mcc, ev_kappa, aucs, df = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_acc.append(1 - ev_test_error)
        all_auc.append(ev_auc_score)
        all_precision.append(ev_precision)
        all_recall.append(ev_recall)
        all_f1.append(ev_f1)
        all_mcc.append(ev_mcc)
        all_kappa.append(ev_kappa)

        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_acc': all_acc, 'test_auc': all_auc, 'test_precision': all_precision,
                             'test_recall': all_recall, 'test_f1': all_f1, 'test_mcc': all_mcc,
                             'test_kappa': all_kappa})
    if len(folds) != args.k:
        save_name = 'ev_summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'ev_summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))

