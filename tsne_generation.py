import os
import sys
import json
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import utils.utils as utils
import utils.config as config
# from train_arcface import train, evaluate_tsne
import modules.base_model_arcface as base_model
from utils.dataset import Dictionary, VQAFeatureDataset
from sklearn.manifold import TSNE
import torch.nn.functional as F


def plot_norms(model, labelnames, y_range=None):
    W = model.weight.cpu()
    print(W.size())
    tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
    name = 'train'
    freq = [0] * 3129
    freq_path = os.path.join(config.cache_root, '{}_freq.json'.format(name))
    qt_dict_freq = json.load(open(freq_path, 'r'))
    for qt_type in qt_dict_freq:
        freq_dic = qt_dict_freq[qt_type]
        freq_dic = {int(k): v for k, v in freq_dic.items()}
        for key in freq_dic:
            freq[key] = freq[key] + freq_dic[key]
    freq = np.array(freq)
    idx = np.argsort(freq)
    idx = idx[::-1]
    freq = freq[idx]
    tmp = tmp[idx]
    print(tmp[0], freq[0], idx[0])
    # for i in range(3129):
    #     print(i, tmp[i])
    if y_range == None:
        max_val, mid_val, min_val = tmp.max(), tmp.mean(), tmp.min()
        print(max_val, mid_val, min_val)
        c = min(1 / mid_val, mid_val)
        y_range = [min_val - c, max_val + c]

    fig = plt.figure(figsize=(15, 3), dpi=64, facecolor='w', edgecolor='k')
    # plt.xticks(list(range(3129)), labelnames, rotation=90, fontsize=8);  # Set text labels.
    ax1 = fig.add_subplot(111)

    ax1.set_ylabel('norm', fontsize=16)
    ax1.set_ylim(y_range)

    plt.plot(tmp, linewidth=2)
    # plt.plot(freq, linewidth=2)
    plt.title('norms of per-class weights from the learned classifier vs. class cardinality', fontsize=20)
    plt.savefig('Norm_plt_normal.png')
    return tmp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of running epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate for adamax')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--num-hid', type=int, default=1024,
                        help='number of dimension in last layer')
    parser.add_argument('--model', type=str, default='baseline_newatt',
                        help='model structure')
    parser.add_argument('--name', type=str, default='exp0.pth',
                        help='saved model name')
    parser.add_argument('--name-new', type=str, default=None,
                        help='combine with fine-tune')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--fine-tune', action='store_true',
                        help='fine tuning with our loss')
    parser.add_argument('--resume', action='store_true',
                        help='whether resume from checkpoint')
    parser.add_argument('--not-save', action='store_true',
                        help='do not overwrite the old model')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='test one time')
    parser.add_argument('--eval-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print('reached')
    args = parse_args()
    print(args)
    print_keys = ['cp_data', 'version', 'train_set',
                  'loss_type', 'use_cos', 'entropy', 'scale']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    seed = 5193
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    dictionary = Dictionary.load_from_file(config.dict_path)
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    args.name = 'logs/' + args.name
    logs = torch.load(args.name)
    print("loading logs from {}".format(args.name))
    constructor = 'build_{}'.format(args.model)
    model, metric_fc = getattr(base_model, constructor)(eval_dset, args.num_hid)
    model = model.cuda()
    metric_fc = metric_fc.cuda()
    model.w_emb.init_embedding(config.glove_embed_path)

    optim = torch.optim.Adamax([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr)

    model.load_state_dict(logs['model_state'])
    metric_fc.load_state_dict(logs['margin_model_state'])
    optim.load_state_dict(logs['optim_state'])
    eval_loader = DataLoader(eval_dset,
                             512, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_dset, 512, shuffle=True, num_workers=4)
    model.eval()
    metric_fc.eval()
    # plot_norms(metric_fc, range(3129))
    with open(os.path.join(config.cache_root, 'train_margin.json')) as f:
        margins = json.load(f)
    print(margins.keys())
    # for qn_type in margins.keys():
    #     evaluate_tsne(model, metric_fc, eval_loader, write=False, tsne=True, qn_type=qn_type)
    plot_norms(metric_fc, "")