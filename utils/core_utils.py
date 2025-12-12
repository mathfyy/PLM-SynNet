from argparse import Namespace
from collections import OrderedDict
import os
import pickle

# from lifelines.utils import concordance_index
import numpy as np
# from sksurv.metrics import concordance_index_censored
from scipy.stats import ttest_ind
import torch
from torch.optim import lr_scheduler

from dataProcess.dataset_generic import save_splits
from utils.utils import *

from utils.train_utils import *

from statsmodels.stats.multitest import multipletests

from models.model_AttnMIL import AttnMIL_Attention, AttnMIL_GatedAttention
from models.model_DeepAttnMISL import DeepAttnMIL
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_TransMIL import TransMIL
from models.model_DTFDMIL import DTFDMIL
from models.model_Surformer import MIL_Attention_FC_Surformer
from models.model_G_HANet import G_HANet
from models.model import LRENet, LRENet_adv


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets: tuple, cur: int, args: Namespace, device):
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda(device.index)
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    # 'AttnMIL', 'AttnMIL_Gate', 'DeepAttnMIL', 'TransMIL', 'DTFDMIL', 'Surformer', 'G_HANet', 'LRENet'
    if args.model == 'AttnMIL':
        model = AttnMIL_Attention(n_classes=args.n_classes, wsi_size=args.embed_dim).to(device)
    elif args.model == 'AttnMIL_Gate':
        model = AttnMIL_GatedAttention(n_classes=args.n_classes, wsi_size=args.embed_dim).to(device)
    elif args.model == 'DeepAttnMIL':
        model = DeepAttnMIL(n_classes=args.n_classes, wsi_size=args.embed_dim).to(device)
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
        model = TransMIL(n_classes=args.n_classes, wsi_size=args.embed_dim).to(device)
    elif args.model == 'DTFDMIL':
        model = DTFDMIL(n_classes=args.n_classes, wsi_size=args.embed_dim).to(device)
    elif args.model == 'Surformer':
        model = MIL_Attention_FC_Surformer(n_classes=args.n_classes, wsi_size=args.embed_dim).to(device)
    elif args.model == 'G_HANet':
        model_dict = {'n_classes': args.n_classes, 'num_tokens': args.num_tokens, 'ratio': args.ratio, 'wsi_size': args.embed_dim}
        model = G_HANet(**model_dict).to(device)
    elif args.model == 'LRENet':
        model_dict = {'n_classes': args.n_classes, 'num_tokens': args.num_tokens, 'ratio': args.ratio, 'wsi_size': args.embed_dim}
        model = LRENet(**model_dict).to(device)
    elif args.model == 'LRENet_adv':
        model_dict = {'n_classes': args.n_classes, 'num_tokens': args.num_tokens, 'ratio': args.ratio, 'wsi_size': args.embed_dim, 'agentDimList': args.agentDimList, 'agentList': args.agentList}
        model = LRENet_adv(**model_dict).to(device)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample,
                                    batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing=args.testing, batch_size=args.batch_size)
    test_loader = get_split_loader(test_split, testing=args.testing, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
        early_stopping_test = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
        early_stopping_test = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, device, args.model)
        stop = validate(cur, epoch, model, val_loader, args.n_classes,
                             early_stopping, writer, loss_fn, args.results_dir, args.model)
        stop_test = validate_test(cur, epoch, model, test_loader, args.n_classes,
                             early_stopping_test, writer, loss_fn, args.results_dir, args.model)
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes, args.model)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, args.model)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


def summary(model, loader, n_classes, modelType):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, data in enumerate(loader):
        slide_id = slide_ids.iloc[batch_idx]
        share_feature = data[0].to(device, non_blocking=True)
        # source_feature = data[1].to(device, non_blocking=True)
        label = data[1].type(torch.LongTensor).to(device)
        with torch.no_grad():
            if modelType == 'LRENet':
                logits, Y_prob, Y_hat, moe_aux_loss, intersection_I, intersection_S, c_I, c_S = model(share_feature,
                                                                                            share_feature)
            elif modelType == 'LRENet_adv':
                logits, Y_prob, Y_hat, moe_aux_loss, kqv, kqv_c = model(share_feature)
            else:
                logits, Y_prob, Y_hat = model(share_feature)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger
