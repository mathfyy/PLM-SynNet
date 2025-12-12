from collections import OrderedDict
from os.path import join

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention_modules import *
from models.foundation_model.conch_model.open_clip_custom import create_model


class Reconstruction_Net(nn.Module):
    def __init__(self, wsi_size=512, num_tokens=6, dropout=0.25):
        super(Reconstruction_Net, self).__init__()
        self.num_tokens = num_tokens

        ### FC Layer over WSI bag
        size = [wsi_size, 512]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(dropout))
        self.wsi_net = nn.Sequential(*fc)

        self.tokens = nn.Parameter(torch.randn((1, self.num_tokens, 512), requires_grad=True))

        self.cross_attention0 = Cross_Attention(query_dim=512, context_dim=512, heads=4, dim_head=128, dropout=dropout)
        self.ffn0 = FeedForward(dim=512, mult=4, dropout=dropout)

        self.cross_attention1 = Cross_Attention(query_dim=512, context_dim=512, heads=4, dim_head=128, dropout=dropout)
        self.ffn1 = FeedForward(dim=512, mult=4, dropout=dropout)

        self.ln_1 = nn.LayerNorm(512)
        self.ln_2 = nn.LayerNorm(512)
        self.ln_3 = nn.LayerNorm(512)
        self.ln_4 = nn.LayerNorm(512)
        self.ln_5 = nn.LayerNorm(512)
        self.ln_6 = nn.LayerNorm(512)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']

        h_path_bag = self.wsi_net(x_path).unsqueeze(0)

        reconstruct, _ = self.cross_attention0(self.ln_1(self.tokens), self.ln_2(h_path_bag))
        reconstruct = self.ffn0(self.ln_3(reconstruct)) + reconstruct

        reconstruct, sim = self.cross_attention0(self.ln_4(self.tokens + reconstruct), self.ln_2(h_path_bag))
        reconstruct = self.ffn1(self.ln_5(reconstruct)) + reconstruct

        reconstruct = self.ln_6(reconstruct)

        return sim, reconstruct


class Hyper_attention(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(Hyper_attention, self).__init__()

        self.linear1 = nn.Linear(in_ft, out_ft)
        self.linear2 = nn.Linear(out_ft, out_ft)

        self.ln1 = nn.LayerNorm(out_ft)
        self.ln2 = nn.LayerNorm(out_ft)
        self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor, scaled_weights=None):
        # x_ori = x

        x_ = G.matmul(x)

        x = self.linear1(x_)
        x = self.ln1(x)
        x = self.relu(x)

        x = self.linear2(x)
        # x = x.matmul(self.weight2) + self.bias2

        x = self.ln2(x)
        x = self.relu(x)
        return x + x_


class G_HANet(nn.Module):
    def __init__(self, n_classes=2, wsi_size=512, dropout=0.25, num_tokens=6, ratio=0.2):
        super(G_HANet, self).__init__()

        self.n_classes = n_classes
        self.num_tokens = num_tokens
        self.ratio = ratio

        ### Gated Layer
        self.gate = Attn_Net_Gated(L=512)

        ### FC Layer over WSI bag
        fc = [nn.Linear(512, wsi_size), nn.LayerNorm(wsi_size), nn.ReLU()]
        fc.append(nn.Dropout(dropout))
        self.wsi_net = nn.Sequential(*fc)

        ### For Reconstruction 
        self.recon_net = Reconstruction_Net(num_tokens=num_tokens)

        ### Classifier
        self.classifier = nn.Linear(num_tokens * 256, n_classes)

        ### Hypergraph layer
        self.hyperconv1 = Hyper_attention(in_ft=wsi_size, out_ft=wsi_size)

        ### MHSA
        self.attention = Self_Attention(query_dim=512, context_dim=512, heads=4, dim_head=128, dropout=dropout)
        self.ffn = FeedForward(dim=512, dropout=dropout)
        self.ln1 = nn.LayerNorm(512)

        self.butter = nn.Sequential(
            nn.Linear(512, 256),
            # nn.LayerNorm(256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        matrix, fea_reconst = self.recon_net(x_path=feature)

        ### for bag transformation
        h_path_bag = self.wsi_net(feature)

        ### generate node-level weights
        weights = self.gate(h_path_bag).T

        ### generate binary matrix
        matrix = matrix.mean(dim=0)
        length = h_path_bag.size(0)

        edge = (matrix >= torch.topk(matrix, dim=1, k=int(length * self.ratio))[0][:, -1].unsqueeze(dim=-1)).float()

        edge1 = (weights)
        edge1 = F.softmax(edge1, dim=1)

        edge2 = (matrix) + (1 - edge) * (-100000)  # (matrix_scaled + weights)/2 * edge
        edge2 = F.softmax(edge2, dim=1).detach()

        fea_hypergraph = self.hyperconv1(h_path_bag, (edge1 + edge2) / 2)

        # fea = torch.cat([fea_hypergraph[None, :, :], fea_reconst], dim=2)

        fea = (fea_hypergraph[None, :, :] + fea_reconst) / 2

        fea = self.attention(fea) + fea
        fea = self.ffn(self.ln1(fea)) + fea

        ### feature compression and flatten
        h = self.butter(fea).view(1, -1)

        ### Survival Layer
        logits = self.classifier(h)  # .unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)

        return logits, hazards, Y_hat
