import os
import numpy as np
import torch
import pickle 
from utils.utils import *
import random
from collections import OrderedDict

from argparse import Namespace
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def samping(patches, labels, p=0.5):
    temp = []
    for i in range(4):
        indexes = torch.nonzero(labels==i)
        length = indexes.size(0)
        num_selected_nodes = int(p * length)
        perm = torch.randperm(length)#.cuda()
        selected_nodes = perm[0: num_selected_nodes]
        
        temp.append(patches[selected_nodes])
    temp = torch.cat(temp, dim=0)
    perm = torch.randperm(temp.size(0)) #.cuda()
    temp = temp[perm]

    return temp

def train_loop(epoch, model, loader, optimizer, n_classes, bag_weight, writer, loss_fn, device, modelType):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    from topk.svm import SmoothTop1SVM
    loss_fn_inst = SmoothTop1SVM(n_classes=n_classes)
    if device.type == 'cuda':
        loss_fn_inst = loss_fn_inst.cuda(device.index)

    if modelType == 'LRENet' or modelType == 'LRENet_adv':
        # loss_fn_mif = torch.nn.SmoothL1Loss()
        loss_fn_mif = torch.nn.L1Loss()
        # loss_fn_mif = torch.nn.MSELoss()

    print('\n')
    for batch_idx, data in enumerate(loader):
        share_feature = data[0].to(device, non_blocking=True)
        label = data[1].type(torch.LongTensor).to(device)

        if modelType == 'LRENet':
            logits, Y_prob, Y_hat, moe_aux_loss, intersection_I, intersection_S, c_I, c_S = model(share_feature, share_feature)
        elif modelType == 'LRENet_adv':
            logits, Y_prob, Y_hat, moe_aux_loss, feature1, feature2 = model(share_feature)
        elif modelType == 'Surformer':
            logits_gl, Y_prob_gl, Y_hat_gl, logits_lo, Y_prob_lo, Y_hat_lo, logits, Y_prob, Y_hat = model(share_feature)
        else:
            logits, Y_prob, Y_hat = model(share_feature)

        acc_logger.log(Y_hat, label)
        if modelType == 'LRENet':
            loss0 = loss_fn(logits, label)
            loss1 = loss_fn_inst(logits, label)
            loss2 = loss_fn_mif(torch.sigmoid(2 * torch.abs(intersection_I - intersection_S)), torch.ones_like(intersection_I)*(torch.tensor(0.5)))
            loss3 = loss_fn_mif(torch.sigmoid(1 / (torch.abs(intersection_I - c_I) + 1)), torch.ones_like(intersection_I)*(torch.tensor(0.5)))
            # loss3 = loss_fn_mif(torch.sigmoid(1/(torch.abs(intersection_I - c_I)+torch.abs(intersection_S - c_S)+1)), torch.zeros_like(intersection_I))
            # loss2 = loss_fn_mif(torch.exp(torch.sigmoid(torch.abs(intersection_I - intersection_S))), torch.ones_like(intersection_I)*torch.exp(torch.tensor(0.5)))
            # loss3 = loss_fn_mif(torch.exp(torch.sigmoid(1/(torch.abs(intersection_I - c_I)+torch.abs(intersection_S - c_S)+1))), torch.ones_like(intersection_I)*torch.exp(torch.tensor(0.5)))
            loss4 = loss_fn_mif(torch.sigmoid(1 / (torch.abs(intersection_S - c_S) + 1)), torch.ones_like(intersection_I)*(torch.tensor(0.5)))

            # loss2 = loss_fn_mif(torch.mm(kqv.t(), kqv_c)/kqv.shape[0], torch.eye(kqv.shape[1]).to(kqv.device))
            # loss3 = loss_fn_mif(torch.sigmoid(1 / (torch.abs(kqv - kqv_c) + 1)), torch.ones_like(kqv)*(torch.tensor(0.5)))
            # for i in range(kqv.shape[0]):
            #     loss2 = loss2 + loss_fn_mif(torch.mm(kqv[i, ].t(), kqv_c[i, ])/kqv.shape[1], torch.eye(kqv.shape[2]).to(kqv.device))
                # loss4 = loss4+loss_fn(kqv_c_pre[i,:,:], label)
            # loss2 = loss2 / kqv.shape[0]
            # loss4 = loss4 / kqv.shape[0]

            # loss0 = loss_fn(logits, label)
            # loss1 = loss_fn_inst(logits, label)
            # loss2 = torch.zeros(1).to(label.device)
            # loss3 = torch.zeros(1).to(label.device)
            # loss4 = torch.zeros(1).to(label.device)
            # for i in range(query_token.shape[0]):
            #     # 特征正交损失
            #     loss2 = loss2 + loss_fn_mif(torch.mm(share_feature_q[i * query_token.shape[1]:(i + 1) * query_token.shape[1], ].t(), share_feature_c[i * query_token.shape[1]:(i + 1) * query_token.shape[1], ])/query_token.shape[1],
            #                                 torch.eye(query_token.shape[2]).to(query_token.device))
            #     # 查询特征差异最大化损失
            #     if query_token.shape[0] > 1:
            #         i_data = share_feature_q[i * query_token.shape[1]:(i + 1) * query_token.shape[1], ]
            #         for j in range(query_token.shape[0]):
            #             if i != j:
            #                 j_data = share_feature_q[j * query_token.shape[1]:(j + 1) * query_token.shape[1], ]
            #                 loss3 = loss3 + loss_fn_mif(1 / (torch.abs(i_data - j_data) + 1), torch.zeros_like(query_token[i]))
            # loss2 = loss2 / query_token.shape[0]
            # if query_token.shape[0] > 1:
            #     loss3 = loss3 / (query_token.shape[0]*(query_token.shape[0]-1))
        elif modelType == 'LRENet_adv':
            use_advLoss1 = True
            use_advLoss2 = True
            weight1 = 2.0
            weight2 = 1.0

            loss0 = loss_fn(logits, label)
            loss1 = loss_fn_inst(logits, label)

            loss2 = torch.zeros(1).to(label.device)
            loss3 = torch.zeros(1).to(label.device)
            loss4 = torch.zeros(1).to(label.device)

            listLoss_q = torch.zeros(feature1.shape[0]).to(label.device)
            listLoss_g = torch.zeros(feature1.shape[0]).to(label.device)
            listLoss_qg = torch.zeros(feature1.shape[0]+1).to(label.device)
            for iPLM in range(feature1.shape[0]):
                logits_q, _, _ = model.OutputLayer(model.GatedSFAttentionBlock(feature1[iPLM, :, :]))
                listLoss_q[iPLM] = bag_weight * loss_fn(logits_q, label) + (1 - bag_weight) * loss_fn_inst(logits_q, label)
                logits_qg, _, _ = model.OutputLayer(model.GatedSFAttentionBlock((feature1[iPLM, :, :]+feature2[iPLM, :, :])/2))
                logits_g, _, _ = model.OutputLayer(model.GatedSFAttentionBlock(feature2[iPLM, :, :]))
                listLoss_qg[iPLM] = bag_weight * loss_fn(logits_qg, label) + (1 - bag_weight) * loss_fn_inst(logits_qg, label)
                listLoss_g[iPLM] = bag_weight * loss_fn(logits_g, label) + (1 - bag_weight) * loss_fn_inst(logits_g, label)
                if use_advLoss1:
                    loss2 = loss2 + listLoss_q[iPLM] + listLoss_g[iPLM]
            if use_advLoss1:
                loss2 = loss2/feature1.shape[0]
            if use_advLoss2:
                listLoss_qg[iPLM+1] = loss0 + loss1
                loss3 = torch.max(torch.max(listLoss_qg) - torch.min(listLoss_q).detach(),
                                  torch.zeros(1).to(label.device))

        elif modelType == 'Surformer':
            loss1 = loss_fn_inst(logits_lo, label)
            loss0 = loss_fn(logits_gl, label) + loss_fn(logits_lo, label) + loss_fn(logits, label)
        else:
            loss0 = loss_fn(logits, label)
            loss1 = loss_fn_inst(logits, label)

        loss_value = loss0.item()
        inst_count += 1
        instance_loss_value = loss1.item()
        train_inst_loss += instance_loss_value
        total_loss = bag_weight * loss0 + (1 - bag_weight) * loss1

        if modelType == 'LRENet':
            loss2_value = loss2.item()
            loss3_value = loss3.item()
            loss4_value = loss4.item()
            total_loss = total_loss + moe_aux_loss + loss2 + loss3 + loss4
        elif modelType == 'LRENet_adv':
            loss2_value = loss2.item()
            loss3_value = loss3.item()
            loss4_value = loss4.item()
            total_loss = total_loss + moe_aux_loss + weight1*loss2 + weight2*loss3

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            if modelType == 'LRENet' or modelType == 'LRENet_adv':
                print('batch {}, loss0: {:.4f}, loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f}, loss4: {:.4f}, loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                      instance_loss_value, loss2_value, loss3_value, loss4_value,
                                                                                                      total_loss.item()) +
                      'label: {}, bag_size: {}'.format(label.item(), data[0].size(0)))
            else:
                print('batch {}, loss0: {:.4f}, loss1: {:.4f}, loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                      instance_loss_value,
                                                                                                      total_loss.item()) +
                      'label: {}, bag_size: {}'.format(label.item(), data[0].size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None, modelType=None):
    from topk.svm import SmoothTop1SVM
    loss_fn_inst = SmoothTop1SVM(n_classes=n_classes)
    if device.type == 'cuda':
        loss_fn_inst = loss_fn_inst.cuda(device.index)

    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0

    logit = np.zeros((len(loader), n_classes))
    hat = np.zeros((len(loader), n_classes))
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.inference_mode():
        for batch_idx, data in enumerate(loader):
            share_feature = data[0].to(device, non_blocking=True)
            # source_feature = data[1].to(device, non_blocking=True)
            label = data[1].type(torch.LongTensor).to(device)

            if modelType == 'LRENet':
                logits, Y_prob, Y_hat, moe_aux_loss, intersection_I, intersection_S, c_I, c_S = model(share_feature,
                                                                                                      share_feature)
            elif modelType == 'LRENet_adv':
                logits, Y_prob, Y_hat, moe_aux_loss, kqv, kqv_c= model(
                    share_feature)
                # if torch.any(torch.isnan(Y_prob)):
                #     cc=0
            else:
                logits, Y_prob, Y_hat = model(share_feature)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = loss_fn_inst(logits, label)
            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            logit[batch_idx] = logits.cpu().numpy()
            hat[batch_idx] = Y_hat.cpu().numpy()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_test(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None, modelType=None):
    from topk.svm import SmoothTop1SVM
    loss_fn_inst = SmoothTop1SVM(n_classes=n_classes)
    if device.type == 'cuda':
        loss_fn_inst = loss_fn_inst.cuda(device.index)

    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.inference_mode():
        for batch_idx, data in enumerate(loader):
            share_feature = data[0].to(device, non_blocking=True)
            # source_feature = data[1].to(device, non_blocking=True)
            label = data[1].type(torch.LongTensor).to(device)

            if modelType == 'LRENet':
                logits, Y_prob, Y_hat, moe_aux_loss, intersection_I, intersection_S, c_I, c_S = model(share_feature,
                                                                                                      share_feature)
            elif modelType == 'LRENet_adv':
                logits, Y_prob, Y_hat, moe_aux_loss, kqv, kqv_c= model(share_feature)
                # if torch.any(torch.isnan(Y_prob)):
                #     cc=0
            else:
                logits, Y_prob, Y_hat = model(share_feature)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = loss_fn_inst(logits, label)
            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\ntest Set, test_loss: {:.4f}, test_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('test/loss', val_loss, epoch)
        writer.add_scalar('test/auc', auc, epoch)
        writer.add_scalar('test/error', val_error, epoch)
        writer.add_scalar('test/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('test/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "test_s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False
