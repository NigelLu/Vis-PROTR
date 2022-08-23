
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from src.model.module.mmn import MMN
from src.model.blocks import DecoderSimple
from src.model.utils import LOSS_DICT

class MMNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.shot = args.shot
        self.norm_s = args.norm_s
        self.norm_q = args.norm_q
        self.att_wt = args.att_wt
        self.im_size = (args.image_size, args.image_size)

        self.inner_loss = LOSS_DICT[args.inner_loss]        # wce loss
        self.meta_loss = LOSS_DICT[args.meta_loss]          # wdc loss

        self.classifier = DecoderSimple(
            n_cls=2,
            d_encoder=args.encoder_dim
        )
        self.trans = MMN(args)

    def meta_params(self):
        return self.trans.parameters()

    @staticmethod
    def compute_weight(label, n_cls):
        try:
            count = torch.bincount(label.flatten())
            weight = torch.tensor([count[0]/count[i] for i in range(n_cls)])
        except:
            weight = torch.ones(n_cls, device=label.device)
        return weight

    def inner_loop(self, f_s, label_s):

        # reset classifier
        self.classifier.reset_parameters()
        self.classifier.train()

        # init weight & optimizer
        weight_s = self.compute_weight(label_s, n_cls=2)
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.lr_cls)

        # adapt the classifier to current task
        for _ in range(self.args.adapt_iter):

            # make prediction
            pred_s = self.classifier(f_s, self.im_size)
            
            # compute loss & update classifier weights
            loss_s = self.inner_loss(pred_s, label_s, weight=weight_s)
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def forward(self, backbone, img_s, img_q, label_s, label_q, use_amp=False):

        # extract feats
        with torch.no_grad():
            f_s, mf_s = backbone.extract_features(img_s)
            f_q, mf_q = backbone.extract_features(img_q)

        # init variables
        pred_q, loss_q = [], []

        # normalize feats as needed
        if self.norm_s:
            f_s = F.normalize(f_s, dim=1)
        if self.norm_q:
            f_q = F.normalize(f_q, dim=1)

        # pred0: inner loop baseline
        self.inner_loop(f_s, label_s)
        self.classifier.eval()
        pred_q.append(self.classifier(f_q, self.im_size))
        loss_q.append(self.meta_loss(pred_q[-1], label_q))

        # automatic mixed precision
        with amp.autocast(enabled=use_amp):

            # pred1: mmn
            att_f_q = []
            for k in range(self.shot):
                curr_f_s = f_s[k:k+1]
                curr_mf_s = {i: f[k:k+1] for i, f in enumerate(mf_s)}
                att_f_q.append(self.trans(mf_q, curr_mf_s, curr_f_s))               # [1, 512, h, w]
            att_f_q = torch.cat(att_f_q, dim=0).mean(dim=0, keepdim=True)           # [k, 512, h, w] => [1, 512, h, w]
            pred_q.append(self.classifier(att_f_q, self.im_size))
            loss_q.append(self.meta_loss(pred_q[-1], label_q))

            # pred2: ensemble
            f_q = (1 - self.att_wt) * f_q + self.att_wt * att_f_q
            pred_q.append(self.classifier(f_q, self.im_size))
            loss_q.append(self.meta_loss(pred_q[-1], label_q))
        
        return pred_q, loss_q