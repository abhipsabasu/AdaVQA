import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import math
import utils.config as config
from modules.fc import FCNet
from modules.attention import Attention, NewAttention
from modules.language_model import WordEmbedding, QuestionEmbedding


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, fusion, num_hid, num_class):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        # self.fusion = fusion
        # # # self.classifier = weight_norm(nn.Linear(num_hid*2, num_class), dim=None)
        # num_hid = num_hid * 2
        # self.kernel = nn.Parameter(torch.Tensor(num_hid, num_class))
        # self.kernel.data.uniform_(-1 / num_hid, 1 / num_hid)

    def forward(self, v, q):
        """
        Forward=

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        # fusion_repr = self.fusion(joint_repr)
        # # logits = self.classifier(logits)
        #
        # if config.use_cos:
        #     k_norm = l2_norm(self.kernel, dim=0)
        #     fusion_repr = l2_norm(fusion_repr, dim=-1)
        # else:
        #     k_norm = self.kernel
        # logits = torch.mm(fusion_repr, k_norm)

        return joint_repr #, logits


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=8.0, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        # self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # self.weight.data.uniform_(-1 / in_features, 1 / in_features)
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        # self.cos_m = math.cos(m)
        # self.sin_m = math.sin(m)
        # self.th = math.cos(math.pi - m)
        # self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label, m):
        m = 1 - m
        self.cos_m = torch.cos(m)
        self.sin_m = torch.sin(m)
        self.th = torch.cos(math.pi - m)
        self.mm = torch.sin(math.pi - m) * m
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # cosine = input
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = (label * phi) + ((1.0 - label) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output = phi * self.s
        # print(output)

        return output, cosine


def l2_norm(input, dim=-1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output


def build_baseline(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)


def build_baseline_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)
    margin_model = ArcMarginProduct(num_hid, dataset.num_ans_candidates)
    return basemodel, margin_model
