from __future__ import print_function

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from dataProcess.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler

from utils.logger import Logger


def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    ### Start 5-Fold CV Evaluation.
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        start = timer()
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path='{}/splits_{}.csv'.format(
                                                                             args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args, device)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

        end = timer()
        print('Fold %d Time: %f seconds' % (i, end - start))

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


### Training settings
parser = argparse.ArgumentParser(description='')
### Checkpoint + Misc. Pathing Parameters
# parser.add_argument('--data_root_dir', type=str,
#                     default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/patch_feature/',
#                     help='Data directory to WSI features (extracted via CLAM')
# parser.add_argument('--data_slide_dir', type=str,
#                     default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/svs/')
# parser.add_argument('--target_patch_size', type=int, default=224)
# parser.add_argument('--split_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/LGG-GBM/splits/task_gbmlgg_100/',
#                     help='Which cancer type within ')
# parser.add_argument('--results_dir', type=str,
#                     default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/result/task_gbmlgg/',
#                     help='Results directory (Default: ./results)')
# parser.add_argument('--exp_code', type=str, default='LRENet_task_gbmlgg', help='experiment code for saving results')
# parser.add_argument('--task', type=str, default='task_gbmlgg', choices=['task_gbmlgg', 'task_pcaepe'])

parser.add_argument('--data_root_dir', type=str,
                    default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_conch/',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--data_slide_dir', type=str, default='/data1/xiamy/PCa-EPE/WSI/')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--split_dir', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/splits/task_pcaepe_100/',
                    help='Which cancer type within ')
parser.add_argument('--results_dir', type=str,
                    default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/result/task_pcaepe/',
                    help='Results directory (Default: ./results)')
parser.add_argument('--exp_code', type=str, default='LRENet_adv_task_pcaepe_wsyn1', help='experiment code for saving results')
parser.add_argument('--task', type=str, default='task_pcaepe', choices=['task_gbmlgg', 'task_pcaepe'])

parser.add_argument('--model', type=str, default='LRENet_adv',
                    choices=['AttnMIL', 'AttnMIL_Gate', 'DeepAttnMIL', 'clam_sb', 'clam_mb', 'TransMIL', 'DTFDMIL', 'Surformer', 'G_HANet', 'LRENet', 'LRENet_adv'])

parser.add_argument('--num_tokens', type=int, default=6, help='Number of tokens ')

# choices=['resnet50_trunc', 'uni_v1', 'uni2-h', 'conch_v1', 'conch_v1_5', 'CHIEF-Ctranspath', 'MUSK']
# # resnet50-1024
# uni-1024
# conch-512
# CHIEF-768
# Virchow-2560
# Virchow2-2560
# prov-gigapath-1536
# uni2-1536
# titan-768
parser.add_argument('--MultiAgent', default=True)
parser.add_argument('--Agent1', type=str, default=None)
parser.add_argument('--Agent1_dim', type=int, default=0)
parser.add_argument('--Agent2', type=str, default=None)
parser.add_argument('--Agent2_dim', type=int, default=0)
# parser.add_argument('--Agent1', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_uni')
# parser.add_argument('--Agent1_dim', type=int, default=1024)
# parser.add_argument('--Agent2', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_Virchow')
# parser.add_argument('--Agent2_dim', type=int, default=2560)
parser.add_argument('--Agent3', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_Virchow2')
parser.add_argument('--Agent3_dim', type=int, default=2560)
parser.add_argument('--Agent4', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_prov-gigapath')
parser.add_argument('--Agent4_dim', type=int, default=1536)
parser.add_argument('--Agent5', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_uni2')
parser.add_argument('--Agent5_dim', type=int, default=1536)
parser.add_argument('--Agent6', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_conch/')
parser.add_argument('--Agent6_dim', type=int, default=512)
parser.add_argument('--Agent7', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_TITAN')
parser.add_argument('--Agent7_dim', type=int, default=768)
parser.add_argument('--Agent8', type=str, default='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/patch_feature_CHIEF/')
parser.add_argument('--Agent8_dim', type=int, default=768)

parser.add_argument('--ratio', type=float, default=0.2, help='For margin loss ')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--which_splits', type=str, default='5foldcv',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--log_data', action='store_true', default=True, help='Log data using tensorboard')

### Model Parameters.
parser.add_argument('--model_size', type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--embed_dim', type=int, default=512)

### Optimizer Parameters
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=True, help='Enable early stopping')

args = parser.parse_args()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_size': args.model_size,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}
print('\nLoad Dataset')

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
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'Dead': 0, 'Alive': 1},
                                  patient_strat=False,
                                  ignore=[])
elif args.task == 'task_pcaepe':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/PCa-EPE_use.csv',
                                  data_dir=os.path.join(args.data_root_dir, ''),
                                  data_slide_dir=os.path.join(args.data_slide_dir, ''),
                                  target_patch_size=args.target_patch_size,
                                  use_h5=False,
                                  MultiAgent=args.MultiAgent,
                                  agentList=agentList,
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'C1': 0, 'C2': 1},
                                  patient_strat=False,
                                  ignore=[])

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

### Sets the absolute path of split_dir
if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    from transformers import AutoModel

    # os.environ["TRANSFORMERS_CACHE"] = "/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/TITAN_model/"
    # from transformers import set_transformers_cache

    # set_transformers_cache("/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/TITAN_model/")
    # cache_dir = '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/TITAN_model/'
    # titan = AutoModel.from_pretrained(
    #     '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/TITAN_model/', cache_dir=cache_dir,
    #     local_files_only=True,
    #     trust_remote_code=False
    # )
    # model, _ = titan.return_conch()

    start = timer()
    sys.stdout = Logger(
        os.path.join(args.results_dir, '/log', str(len(os.listdir(args.results_dir))) + '/log_' + args.split_dir.split('/')[-1] + '.txt'))
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
