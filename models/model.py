from collections import OrderedDict
from os.path import join

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.attention_modules import *
from models.FANLayer import FANLayer
from models.foundation_model.conch_model.open_clip_custom import create_model
from tutel import moe as tutel_moe
from sympy.matrices import Matrix, GramSchmidt
from timm.models.layers import DropPath


global_var = 0

def change_function():
    global global_var
    global_var = global_var+1
    return


class genQureyToken(nn.Module):
    def __init__(self, model_dim=512, dropout=0.25):
        super().__init__()
        self.gen_encodeToken = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.LayerNorm(model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gen_GlobalTokenInformation = nn.AdaptiveAvgPool1d(1)

        self.gen_decodeComplementaryToken = nn.Sequential(
            nn.Linear(model_dim // 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, FeatureExtractTokens):
        x = self.gen_GlobalTokenInformation(self.gen_encodeToken(FeatureExtractTokens).t())
        complementaryToken = self.gen_decodeComplementaryToken(x.expand(-1, FeatureExtractTokens.shape[0]).t())
        return complementaryToken


def orthogo_tensor(x):
    m, n = x.size()
    x_np = x.t().cpu().detach().numpy()
    matrix = [Matrix(col) for col in x_np.T]
    gram = GramSchmidt(matrix)
    ort_list = []
    for i in range(m):
        vector = []
        for j in range(n):
            vector.append(float(gram[i][j]))
        ort_list.append(vector)
    ort_list = np.mat(ort_list)
    ort_list = torch.from_numpy(ort_list)
    ort_list = F.normalize(ort_list, dim=1)
    return ort_list.to(x.device).type_as(x)


def schmidt_orthogonalization(matrix):
    m, n = matrix.shape
    Q = torch.zeros_like(matrix)
    for i in range(m):
        # 取第i个行向量
        q = matrix[i].clone()
        # 减去前i-1个正交向量的投影
        for j in range(i):
            qj = Q[j]
            proj = torch.dot(q, qj) / torch.dot(qj, qj)
            q = q - proj * qj
        # 单位化
        q = q / torch.norm(q)
        Q[i] = q
    return Q


class IndependentFeatureExtractBlock(nn.Module):
    def __init__(self, model_dim=512, dropout=0.5):
        super().__init__()

        self.ln1 = nn.LayerNorm(model_dim)

        self.fc1_q = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.fc1_k = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )
        self.fc1_v = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.ln2 = nn.LayerNorm(model_dim)

        self.fc2_q = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.fc2_k = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )
        self.fc2_v = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=-1)

        self.genQureyToken_share = genQureyToken(model_dim)

        self.genQureyToken_specific = genQureyToken(model_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.mlp_I_kqv = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.mlp_S_kqv = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.mlp_c_I_kqv = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.mlp_c_S_kqv = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

    def forward(self, specific_feature, share_feature, FeatureExtractTokens):
        specific_feature = self.ln1(specific_feature)
        share_feature = self.ln2(share_feature)
        d_k = FeatureExtractTokens.shape[1]
        Q_specific = self.fc1_q(FeatureExtractTokens)
        K_specific = self.fc1_k(specific_feature)
        V_specific = self.fc1_v(specific_feature)
        I_kq = torch.matmul(K_specific, Q_specific.permute(1, 0)) / d_k
        I_kqv = torch.matmul(V_specific.permute(1, 0), self.dropout1(self.softmax(I_kq))).permute(1, 0) / d_k

        Q_share = self.fc2_q(FeatureExtractTokens)
        K_share = self.fc2_k(share_feature)
        V_share = self.fc2_v(share_feature)
        S_kq = torch.matmul(K_share, Q_share.permute(1, 0)) / d_k
        S_kqv = torch.matmul(V_share.permute(1, 0), self.dropout2(self.softmax(S_kq))).permute(1, 0) / d_k
        # S_kqv_O = orthogo_tensor(S_kqv)
        # 生成specific_feature和share_feature的互补查询Token
        complementaryToken_specific = self.genQureyToken_specific(Q_specific)
        complementaryToken_share = self.genQureyToken_share(Q_share)

        c_I_kq = torch.matmul(K_specific, complementaryToken_specific.permute(1, 0)) / d_k
        c_I_kqv = torch.matmul(V_specific.permute(1, 0), self.dropout1(self.softmax(c_I_kq))).permute(1, 0) / d_k

        c_S_kq = torch.matmul(K_share, complementaryToken_share.permute(1, 0)) / d_k
        c_S_kqv = torch.matmul(V_share.permute(1, 0), self.dropout1(self.softmax(c_S_kq))).permute(1, 0) / d_k

        return self.mlp_c_I_kqv(c_I_kqv), self.mlp_c_S_kqv(c_S_kqv), self.mlp_I_kqv(I_kqv), self.mlp_S_kqv(S_kqv)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x_new: torch.Tensor, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x_new, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x_new: torch.Tensor, x: torch.Tensor):
        x_new = x_new + self.attention(self.ln_1(x_new), self.ln_1(x))
        x_new = x_new + self.mlp(self.ln_2(x_new))
        return x_new


class GmoeExpert(nn.Module):
    def __init__(self, model_dim=512, local_experts=1,
                 sharded_count=1,
                 my_config=None, dropout=0.25):
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            # ("c_fc", nn.Linear(model_dim, model_dim)),
            # ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(model_dim, model_dim))
        ]))

    def forward(self, feature, ctx):
        if ctx.sharded_count > 1:
            raise Exception("`sharded_count > 1` is not implemented within this expert, Model parallel is disabled.")
        gated_feature = self.mlp(feature)
        return gated_feature


class GatedSFAttentionBlock0(nn.Module):
    def __init__(self, embeding_dim=512, dropout=0.25):
        super(GatedSFAttentionBlock0, self).__init__()
        self.attn_s = nn.MultiheadAttention(embeding_dim, embeding_dim // 64)
        self.attn_fr = nn.MultiheadAttention(embeding_dim, embeding_dim // 64)
        self.attn_fi = nn.MultiheadAttention(embeding_dim, embeding_dim // 64)

        self.ln_r = LayerNorm(embeding_dim)
        self.ln_i = LayerNorm(embeding_dim)

        self.ln_1 = LayerNorm(embeding_dim)

        self.ln_2 = LayerNorm(embeding_dim)
        # self.FANLayer = FANLayer(embeding_dim, embeding_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embeding_dim, embeding_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(embeding_dim * 4, embeding_dim))
        ]))

        # num_local_experts = 6
        # dist_rank = 1
        # self.mlp = tutel_moe.moe_layer(
        #     gate_type={'type': 'cosine_top', 'k': 1, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.0},
        #     experts={'type': 'custom', 'module': GmoeExpert, 'count_per_node': num_local_experts,
        #              'my_config': None},
        #     model_dim=embeding_dim, scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
        #     seeds=(1, dist_rank + 1, 1), a2a_ffn_overlap_degree=1, use_2dh=False, batch_prioritized_routing=True,
        #     is_gshard_loss=False,
        # )
        # self.aux_loss_weights = 0.01

    def forward(self, feature):
        x_ft = torch.fft.fft2(feature)
        x_real = self.attn_fr(self.ln_r(x_ft.real), self.ln_r(x_ft.real), self.ln_r(x_ft.real))[0]
        x_imag = self.attn_fi(self.ln_i(x_ft.imag), self.ln_i(x_ft.imag), self.ln_i(x_ft.imag))[0]
        x_ft_new = torch.view_as_complex(torch.stack([x_real - x_imag, x_real + x_imag], dim=-1))
        x_new = feature + self.attn_s(self.ln_1(feature), self.ln_1(feature), self.ln_1(feature))[0] + torch.fft.ifft2(
            x_ft_new).real

        x_out = x_new + self.mlp(self.ln_2(x_new))
        # moe_aux_loss = 0.0
        moe_aux_loss = self.mlp.l_aux * self.aux_loss_weights
        return x_out, moe_aux_loss


class GatedSFAttentionBlock(nn.Module):
    def __init__(self, embeding_dim=512, dropout=0.25):
        super(GatedSFAttentionBlock, self).__init__()
        self.attn_s = nn.MultiheadAttention(embeding_dim, embeding_dim // 64)
        self.ln_1 = LayerNorm(embeding_dim)
        self.ln_2 = LayerNorm(embeding_dim)
        # self.FANLayer = FANLayer(embeding_dim, embeding_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embeding_dim, embeding_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(embeding_dim * 4, embeding_dim))
        ]))
        self.drop_path = DropPath(0.0) if dropout > 0. else nn.Identity()

    def forward(self, feature):
        # feature = torch.cat((feature, self.FANLayer(feature)), dim=0)
        x_new = feature + self.drop_path(self.attn_s(self.ln_1(feature), self.ln_1(feature), self.ln_1(feature))[0])
        x_out = x_new + self.drop_path(self.mlp(self.ln_2(x_new)))

        return x_out


class OutputBlock(nn.Module):
    def __init__(self, wsi_size: int, dropout: float, num_tokens: int, ratio: float, n_classes: int):
        super().__init__()

        ### Classifier
        self.classifier = nn.Linear(num_tokens * 256, n_classes)

        self.butter = nn.Sequential(
            nn.Linear(wsi_size, 256)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature: torch.Tensor):
        ### feature compression and flatten
        h = self.butter(feature).view(1, -1)

        logits = self.classifier(h)  # .unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        return logits, hazards, Y_hat


class LRENet(nn.Module):
    def __init__(self, n_classes=2, wsi_size=512, dropout=0.25, num_tokens=5, ratio=0.2,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(LRENet, self).__init__()

        self.n_classes = n_classes
        self.num_tokens = num_tokens

        # self.basebone = create_model("conch_ViT-B-16",
        #                              checkpoint_path='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/CONCH_model/pytorch_model.bin',
        #                              device=device)
        # from functools import partial
        # self.basebone.forward = partial(self.basebone.encode_image, proj_contrast=False, normalize=False)

        self.layers_num = 3
        self.FeatureExtractTokens = nn.Parameter(
            wsi_size ** -0.5 * torch.randn((self.layers_num, self.num_tokens, wsi_size)))

        self.Indblocks = nn.Sequential(
            *[IndependentFeatureExtractBlock(wsi_size, dropout) for _ in range(self.layers_num)])

        self.GatedSFAttentionBlock = GatedSFAttentionBlock(wsi_size, dropout)

        self.OutputLayer = OutputBlock(wsi_size, dropout, num_tokens * (2 + self.layers_num), ratio, n_classes)

    def forward(self, share_feature, source_feature):
        c_I, c_S, intersection_I, intersection_S = self.Indblocks[0](source_feature, share_feature,
                                                                     self.FeatureExtractTokens[0])
        for i in range(self.layers_num - 1):
            c_I_i, c_S_i, intersection_I_i, intersection_S_i = self.Indblocks[i + 1](
                c_I[i * self.num_tokens:(i + 1) * self.num_tokens], c_S[i * self.num_tokens:(i + 1) * self.num_tokens],
                self.FeatureExtractTokens[i + 1])
            intersection_I = torch.cat((intersection_I, intersection_I_i), dim=0)
            intersection_S = torch.cat((intersection_S, intersection_S_i), dim=0)
            c_I = torch.cat((c_I, c_I_i), dim=0)
            c_S = torch.cat((c_S, c_S_i), dim=0)

        # 有用特征拼接
        fusion_feature = torch.cat((c_S[(self.layers_num - 1) * self.num_tokens:self.layers_num * self.num_tokens],
                                    intersection_S,
                                    c_I[(self.layers_num - 1) * self.num_tokens:self.layers_num * self.num_tokens]),
                                   dim=0)
        # 空间频域注意力融合
        # gated_feature, moe_aux_loss = self.GatedSFAttentionBlock(fusion_feature)
        moe_aux_loss = 0.0

        logits, hazards, Y_hat = self.OutputLayer(fusion_feature)
        # 1.+2个fc层，增加c_I与分类任务的相关性
        return logits, hazards, Y_hat, moe_aux_loss, intersection_I, intersection_S, c_I, c_S


class ShareFeatureSplitComplementBlock(nn.Module):
    def __init__(self, model_dim=512, dropout=0.5):
        super().__init__()

        self.fc_q = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim, bias=False),
            # nn.Tanh()
        )

        self.ln = nn.LayerNorm(model_dim)
        self.fc_k = nn.Sequential(
            nn.Linear(model_dim, model_dim, bias=False),
            # nn.Tanh()
        )
        self.fc_v = nn.Sequential(
            nn.Linear(model_dim, model_dim, bias=False),
            # nn.Tanh()
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.genQureyToken = genQureyToken(model_dim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, share_feature, FeatureExtractTokens):
        share_feature = self.ln(share_feature)
        d_k = FeatureExtractTokens.shape[1]
        Q_share = self.fc_q(FeatureExtractTokens)
        K_share = self.fc_k(share_feature)
        V_share = self.fc_v(share_feature)
        S_kq = torch.matmul(K_share, Q_share.permute(1, 0)) / d_k
        # share_feature_q = torch.matmul(V_share.permute(1, 0), self.dropout(self.softmax(S_kq))).permute(1, 0) / d_k
        share_feature_q = self.mlp(
            torch.matmul(V_share.permute(1, 0), self.dropout(self.softmax(S_kq))).permute(1, 0) / d_k)

        complementaryToken = self.genQureyToken(Q_share)
        c_S_kq = torch.matmul(K_share, complementaryToken.permute(1, 0)) / d_k
        # share_feature_q_c = torch.matmul(V_share.permute(1, 0), self.dropout1(self.softmax(c_S_kq))).permute(1, 0) / d_k
        share_feature_q_c = self.mlp_c(
            torch.matmul(V_share.permute(1, 0), self.dropout1(self.softmax(c_S_kq))).permute(1, 0) / d_k)

        return share_feature_q, share_feature_q_c, complementaryToken.unsqueeze(0)


class ShareFeatureSplitComplementBlock1(nn.Module):
    def __init__(self, model_dim=512, dropout=0.5):
        super().__init__()

        self.fc_q = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim, bias=False),
            # nn.Tanh()
        )

        self.ln = nn.LayerNorm(model_dim)
        self.fc_k = nn.Sequential(
            nn.Linear(model_dim, model_dim, bias=False),
            # nn.Tanh()
        )
        self.fc_v = nn.Sequential(
            nn.Linear(model_dim, model_dim, bias=False),
            # nn.Tanh()
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            # nn.Tanh()
        )

        self.genTokenComplement = genQureyToken(model_dim)
        self.dropout1 = nn.Dropout(dropout)

    def subSpaceFeatureRetrieval(self, Q, K, V):
        d_k = Q.shape[0]
        kq = torch.matmul(K, Q.permute(1, 0)) / d_k
        # kqv = torch.matmul(V_share.permute(1, 0), self.dropout(self.softmax(S_kq))).permute(1, 0) / d_k
        kqv = self.mlp(
            torch.matmul(V.permute(1, 0), self.dropout(self.softmax(kq))).permute(1, 0) / d_k)
        return kqv

    def forward(self, share_feature, FeatureExtractTokens):
        share_feature = self.ln(share_feature)
        Q_share = self.fc_q(FeatureExtractTokens)
        K_share = self.fc_k(share_feature)
        V_share = self.fc_v(share_feature)
        kqv = torch.zeros_like(FeatureExtractTokens.reshape((-1, FeatureExtractTokens.shape[2])))
        kqv_c = torch.zeros_like(FeatureExtractTokens.reshape((-1, FeatureExtractTokens.shape[2])))
        # Tokens_c = torch.zeros_like(FeatureExtractTokens)
        for i in range(FeatureExtractTokens.shape[0]):
            kqv_i = self.subSpaceFeatureRetrieval(Q_share[i, :, :], K_share, V_share)
            kqv[i * FeatureExtractTokens.shape[1]:(i + 1) * FeatureExtractTokens.shape[1], :] = kqv_i
            # complementaryToken_i = self.genQureyToken(Q_share[i, :, :])
            # Tokens_c[i, :, :] = complementaryToken_i
            # kqv_c_i = self.subSpaceFeatureRetrieval(complementaryToken_i, -K_share, -V_share)
            kqv_c[i * FeatureExtractTokens.shape[1]:(i + 1) * FeatureExtractTokens.shape[1],
            :] = self.genTokenComplement(kqv_i)

        return kqv, FeatureExtractTokens, kqv_c


class LRENet_1(nn.Module):
    def __init__(self, n_classes=2, wsi_size=512, dropout=0.25, num_tokens=5, ratio=0.2,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(LRENet_1, self).__init__()

        self.n_classes = n_classes
        self.num_tokens = num_tokens

        # self.basebone = create_model("conch_ViT-B-16",
        #                              checkpoint_path='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/CONCH_model/pytorch_model.bin',
        #                              device=device)
        # from functools import partial
        # self.basebone.forward = partial(self.basebone.encode_image, proj_contrast=False, normalize=False)

        self.layers_num = 1
        self.FeatureExtractTokens = nn.Parameter(
            wsi_size ** -0.5 * torch.randn((self.layers_num, self.num_tokens, wsi_size)))

        # self.Indblocks = nn.Sequential(
        #     *[ShareFeatureSplitComplementBlock(wsi_size, dropout) for _ in range(self.layers_num)])
        self.Indblocks = ShareFeatureSplitComplementBlock1(wsi_size, dropout)

        self.GatedSFAttentionBlock = GatedSFAttentionBlock(wsi_size, dropout)

        self.OutputLayer = OutputBlock(wsi_size, dropout, num_tokens * 4 * self.layers_num, ratio, n_classes)

    def forward(self, share_feature):
        kqv, Tokens_c, kqv_c = self.Indblocks(share_feature, self.FeatureExtractTokens)
        # share_feature_q, share_feature_c, complementaryToken = self.Indblocks[0](share_feature, self.FeatureExtractTokens[0])
        # for i in range(self.layers_num - 1):
        #     share_feature_i_q, share_feature_i_c, complementaryToken_i = self.Indblocks[i + 1](share_feature,
        #                                                                  self.FeatureExtractTokens[i + 1])
        #     share_feature_q = torch.cat((share_feature_q, share_feature_i_q), dim=0)
        #     share_feature_c = torch.cat((share_feature_c, share_feature_i_c), dim=0)
        #     complementaryToken = torch.cat((complementaryToken, complementaryToken_i), dim=0)

        # 有用特征拼接
        fusion_feature = torch.cat((kqv, kqv_c), dim=0)
        # 空间频域注意力融合
        gated_feature, moe_aux_loss = self.GatedSFAttentionBlock(fusion_feature)
        # moe_aux_loss = 0.0

        logits, hazards, Y_hat = self.OutputLayer(gated_feature)
        # 1.+2个fc层，增加c_I与分类任务的相关性
        return logits, hazards, Y_hat, moe_aux_loss, kqv, kqv_c, self.FeatureExtractTokens, Tokens_c


class QueryKeyFeature(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, dropout=0.5):
        super().__init__()

        # self.fc_q = nn.Sequential(
        #     nn.LayerNorm(in_dim),
        #     nn.Linear(in_dim, in_dim, bias=False),
        #     # nn.Tanh()
        # )

        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)
        # self.fc_k = nn.Sequential(
        #     nn.Linear(in_dim, in_dim, bias=False),
        #     # nn.Tanh()
        # )
        # self.fc_v = nn.Sequential(
        #     nn.Linear(in_dim, in_dim, bias=False),
        #     # nn.Tanh()
        # )

        # self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.Tanh()
        )
        self.attn = nn.MultiheadAttention(in_dim, in_dim)

    def subSpaceFeatureRetrieval(self, Q, K, V):
        value, weight = self.attn(Q, K, V, need_weights=True, attn_mask=None)

        # d_k = Q.shape[1]
        # kq = torch.matmul(K, Q.permute(1, 0)) / d_k
        # kqv = self.mlp(
        #     torch.matmul(V.permute(1, 0), self.dropout(self.softmax(kq))).permute(1, 0) / d_k)
        # print query attn
        # change_function()
        # savePath = r'/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/save_/' + str(global_var) + '_' + str(weight.shape[1]) + '_atte.mat'
        # from scipy.io import savemat
        # attn_weight = (weight-weight.min())/(weight.max()-weight.min())
        # savemat( savePath , {'data': attn_weight.mean(0).cpu().numpy()})

        return self.mlp(value)

    def forward(self, share_feature, FeatureExtractTokens):
        share_feature = self.ln1(share_feature)
        FeatureExtractTokens = self.ln2(FeatureExtractTokens)
        # Q_share = self.fc_q(FeatureExtractTokens)
        # K_share = self.fc_k(share_feature)
        # V_share = self.fc_v(share_feature)
        kqv = self.subSpaceFeatureRetrieval(FeatureExtractTokens, share_feature, share_feature)
        return kqv


class FeatureComplementBlock(nn.Module):
    def __init__(self, model_dim=512, dropout=0.25):
        super().__init__()
        self.gen_encodeToken = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.LayerNorm(model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gen_GlobalTokenInformation = nn.AdaptiveAvgPool1d(1)

        self.gen_decodeComplementaryToken = nn.Sequential(
            nn.Linear(model_dim // 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, FeatureExtractTokens):
        x = self.gen_GlobalTokenInformation(self.gen_encodeToken(FeatureExtractTokens).t())
        complementaryToken = self.gen_decodeComplementaryToken(x.expand(-1, FeatureExtractTokens.shape[0]).t())
        return complementaryToken


class LRENet_2(nn.Module):
    def __init__(self, n_classes=2, wsi_size=512, dropout=0.25, num_tokens=5, ratio=0.2, agentDimList=list([]),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(LRENet_2, self).__init__()

        self.n_classes = n_classes
        self.num_tokens = num_tokens
        self.agentDimList = agentDimList
        self.wsi_size = wsi_size
        # self.basebone = create_model("conch_ViT-B-16",
        #                              checkpoint_path='/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/pretrain_model/CONCH_model/pytorch_model.bin',
        #                              device=device)
        # from functools import partial
        # self.basebone.forward = partial(self.basebone.encode_image, proj_contrast=False, normalize=False)

        self.layers_num = len(agentDimList)
        self.FeatureExtractTokens = nn.ParameterList(
            [nn.Parameter((agentDimList[i] ** -0.5) * torch.randn(self.num_tokens, agentDimList[i])) for i in
             range(self.layers_num)])
        self.Indblocks = nn.Sequential(
            *[QueryKeyFeature(agentDimList[i], wsi_size, dropout) for i in range(self.layers_num)])
        self.lamda = nn.Parameter(torch.rand((self.layers_num, 1)))
        # self.FeatureComplementBlock = FeatureComplementBlock(wsi_size)

        num_local_experts = self.layers_num
        self.moe_drop = nn.Dropout(0.1)
        self.moemlp = tutel_moe.moe_layer(
            gate_type={'type': 'cosine_top', 'k': 1, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.0},
            experts={'type': 'ffn', 'count_per_node': num_local_experts,
                     'hidden_size_per_expert': wsi_size * 4,
                     'activation_fn': lambda x: self.moe_drop(F.gelu(x))},
            model_dim=wsi_size,
            batch_prioritized_routing=True,
            is_gshard_loss=False,
        )
        self.aux_loss_weights = 0.01

        self.GatedSFAttentionBlock = GatedSFAttentionBlock(wsi_size, dropout)

        self.OutputLayer = OutputBlock(wsi_size, dropout, self.num_tokens, ratio, n_classes)
        # self.classifier = nn.Linear(num_tokens * wsi_size, n_classes)

    def forward(self, share_feature):
        # multi agent feature fusion and complement
        moe_aux_loss = torch.zeros(1).to(share_feature.device)
        cur_feature = torch.zeros(size=(self.num_tokens, self.wsi_size)).to(share_feature.device)
        feature1 = torch.zeros(size=(self.layers_num, self.num_tokens, self.wsi_size)).to(share_feature.device)
        feature2 = torch.zeros(size=(self.layers_num, self.num_tokens, self.wsi_size)).to(share_feature.device)
        feature2_pre = torch.zeros(size=(self.layers_num, 1, self.n_classes)).to(share_feature.device)
        for i in range(self.layers_num):
            b_loc = torch.tensor(self.agentDimList)[:i].sum()
            e_loc = b_loc + self.agentDimList[i]
            query_feature = self.Indblocks[i](share_feature[:, b_loc:e_loc], self.FeatureExtractTokens[i])
            cur_feature = cur_feature + self.moemlp(query_feature)
            moe_aux_loss = moe_aux_loss + self.moemlp.l_aux * self.aux_loss_weights
            # cur_feature[i*self.num_tokens:(i+1)*self.num_tokens, :] = query_feature
            # cur_feature = cur_feature + (self.lamda[i]/self.lamda.sum()) * query_feature

            feature1[i, :, :] = cur_feature
            # comp_feature = self.FeatureComplementBlock(cur_feature)
            # feature2[i, :, :] = comp_feature
            # feature2_pre[i, :, :] = self.classifier(comp_feature.view(1, -1))
        # cur_feature = cur_feature + self.moemlp(cur_feature)
        # moe_aux_loss = self.moemlp.l_aux * self.aux_loss_weights
        # comp_feature = self.FeatureComplementBlock(cur_feature)
        cur_feature = cur_feature / self.layers_num
        # cat all feature
        # fusion_feature = torch.cat((cur_feature, comp_feature), dim=0)

        # MH-Atte
        gated_feature, moe_aux_loss1 = self.GatedSFAttentionBlock(cur_feature)
        # moe_aux_loss = 0.0

        logits, hazards, Y_hat = self.OutputLayer(gated_feature)

        return logits, hazards, Y_hat, moe_aux_loss, feature1, feature2, feature2_pre


class FeatureComplementRxpert(nn.Module):
    def __init__(self, model_dim=512, dropout=0.25, local_experts=1,
                 sharded_count=1,
                 my_config=None):
        super().__init__()
        self.gen_encodeToken = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.LayerNorm(model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gen_GlobalTokenInformation = nn.AdaptiveAvgPool1d(1)

        self.gen_decodeComplementaryToken = nn.Sequential(
            nn.Linear(model_dim // 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, FeatureToken, ctx):
        if ctx.sharded_count > 1:
            raise Exception("`sharded_count > 1` is not implemented within this expert, Model parallel is disabled.")
        FeatureToken_e = self.gen_encodeToken(FeatureToken)
        FeatureToken_mid = FeatureToken_e + self.gen_GlobalTokenInformation(FeatureToken_e).expand(FeatureToken_e.shape[0], FeatureToken_e.shape[1], FeatureToken_e.shape[2])
        complementaryToken = self.gen_decodeComplementaryToken(FeatureToken_mid)
        return complementaryToken

class LRENet_adv(nn.Module):
    def __init__(self, n_classes=2, wsi_size=512, dropout=0.25, num_tokens=5, ratio=0.2, agentDimList=list([]), agentList=list([]),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(LRENet_adv, self).__init__()

        self.n_classes = n_classes
        self.num_tokens = num_tokens
        self.agentList = agentList
        self.agentDimList = agentDimList
        self.wsi_size = wsi_size

        self.layers_num = len(agentDimList)
        if self.layers_num > 0:
            self.FeatureExtractTokens = nn.ParameterList(
                [nn.Parameter((agentDimList[i] ** -0.5) * torch.randn(self.num_tokens, agentDimList[i])) for i in
                 range(self.layers_num)])
            self.Indblocks = nn.Sequential(
                *[QueryKeyFeature(agentDimList[i], wsi_size, dropout) for i in range(self.layers_num)])

            num_local_experts = 9
            self.moemlp = tutel_moe.moe_layer(
                gate_type={'type': 'cosine_top', 'k': 1, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.0},
                experts={'type': 'custom', 'module': FeatureComplementRxpert, 'count_per_node': num_local_experts,
                         'my_config': None},
                model_dim=wsi_size, scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
                seeds=(1, 1 + 1, 1), a2a_ffn_overlap_degree=1, use_2dh=False, batch_prioritized_routing=True,
                is_gshard_loss=False,
            )
            self.aux_loss_weights = 0.01
            # moe_layer visual
            # map_image = crit[1][0].unsqueeze(1).view(list(original_shape[:-reserve_dims]))[:,1:].to(original_dtype)
            # map_image_ = torch.nn.functional.interpolate(map_image.reshape(map_image.shape[0], 1, 16, 8), scale_factor=16,mode='nearest')
            # import SimpleITK as sitk
            # sitk.WriteImage(sitk.GetImageFromArray(map_image_.squeeze().to("cpu").detach().numpy()), os.path.join(r'/data/object/Out_MFRNet/img/', 'map.nii'))
        else:
            self.FeatureExtractTokens = nn.Parameter((wsi_size ** -0.5) * torch.randn(self.num_tokens, wsi_size))
            self.Indblocks = QueryKeyFeature(wsi_size, wsi_size, dropout)

        self.GatedSFAttentionBlock = GatedSFAttentionBlock(wsi_size, dropout)
        self.OutputLayer = OutputBlock(wsi_size, dropout, self.num_tokens, ratio, n_classes)

    def forward(self, share_feature):
        agentList_test = self.agentList
        if self.training is False:
            agentList_test = list([])
            for i in range(self.layers_num):
                if i >= 0:
                    agentList_test.append(self.agentList[i])
                else:
                    agentList_test.append(None)
        moe_aux_loss = torch.zeros(1).to(share_feature.device)
        feature1 = torch.tensor([]).to(share_feature.device)
        feature2 = torch.tensor([]).to(share_feature.device)
        if self.layers_num > 0:
            # multi agent feature fusion and complement
            num = torch.zeros(1).to(share_feature.device)
            for i in range(self.layers_num):
                if agentList_test[i] is not None:
                    b_loc = torch.tensor(self.agentDimList)[:i].sum()
                    e_loc = b_loc + self.agentDimList[i]
                    query_feature = self.Indblocks[i](share_feature[:, b_loc:e_loc], self.FeatureExtractTokens[i])
                    gen_feature = self.moemlp(query_feature)
                    moe_aux_loss = moe_aux_loss + self.moemlp.l_aux * self.aux_loss_weights
                    feature1 = torch.cat((feature1, query_feature.unsqueeze(0)), dim=0)
                    feature2 = torch.cat((feature2, gen_feature.unsqueeze(0)), dim=0)
                    num = num+1
            cur_feature = (feature1 + feature2).sum(dim=0)/(2*num)
            # cur_feature = feature1.sum(dim=0) / num
        else:
            cur_feature = self.Indblocks(share_feature, self.FeatureExtractTokens)

        logits, hazards, Y_hat = self.OutputLayer(self.GatedSFAttentionBlock(cur_feature))

        return logits, hazards, Y_hat, moe_aux_loss, feature1, feature2
