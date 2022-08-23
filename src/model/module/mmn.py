
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.module.match import MatchNet
from src.model.utils import get_corr

class MMN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wa = args.wa               # True or False
        self.agg = args.agg             # 'cat' | 'sum'
        self.rmid = args.rmid           # List[int]
        self.red_dim = args.red_dim     # int or False
        self.feature_channels = [128, 256, 512, 1024, 2048]

        if self.wa or (self.red_dim != False):
            for bid in self.rmid:
                c_in, c_out = self.feature_channels[bid], self.red_dim
                if isinstance(self.red_dim, int) and self.red_dim != False:
                    setattr(self, "rd_" + str(bid), nn.Sequential(
                        nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.ReLU(inplace=True)
                    ))
                    c_in = self.red_dim
                setattr(self, "wa_" + str(bid), WeightAverage(c_in, args))
        
        self.corr_net = MatchNet(
            temp=args.temp,
            cv_type=args.get('conv4d', 'red'),
            in_channel=(1 if self.agg == 'sum' else len(self.rmid)),
            sce=False,
            cyc=False,
            sym_mode=True
        )

    def forward(self, mf_q, mf_s, f_s):
        
        B, _, H, W = f_s.shape

        corr_lst = []
        for idx in self.rmid:
            fq_fea, fs_fea = mf_q[idx], mf_s[idx]               # [B, C, H, W]
            if self.red_dim:
                fq_fea = getattr(self, 'rd_'+str(idx))(fq_fea)
                fs_fea = getattr(self, 'rd_'+str(idx))(fs_fea)
            if self.wa:
                fq_fea = getattr(self, "wa_"+str(idx))(fq_fea)
                fs_fea = getattr(self, 'wa_'+str(idx))(fs_fea)
            corr = get_corr(fq_fea, fs_fea)                     # [B, N_q, N_s]
            corr = corr.view(B, -1, H, W, H, W)
            corr_lst.append(corr)

        corr4d = torch.cat(corr_lst, dim=1)                     # [B, L, H, W, H, W]
        if self.agg == 'sum':
            corr4d = torch.sum(corr4d, dim=1, keepdim=True)     # [B, 1, H, W, H, W]

        att_fq = self.corr_net.corr_forward(corr4d, v=f_s)      # [shot, H, W]
        att_fq = att_fq.mean(dim=0, keepdim=True)

        return att_fq

class WeightAverage(nn.Module):
    def __init__(self, c_in, args, R=3):
        super(WeightAverage, self).__init__()
        c_out = c_in // 2

        self.conv_theta = nn.Conv2d(c_in, c_out, 1)
        self.conv_phi = nn.Conv2d(c_in, c_out, 1)
        self.conv_g = nn.Conv2d(c_in, c_out, 1)
        self.conv_back = nn.Conv2d(c_out, c_in, 1)
        self.CosSimLayer = nn.CosineSimilarity(dim=3)  # norm

        self.R = R
        self.c_out = c_out
        self.att_drop = nn.Dropout(args.get('att_drop', 0.0))
        self.proj_drop = nn.Dropout(args.get('proj_drop', 0.0))

    def forward(self, x):
        """
        x: torch.Tensor(batch_size, channel, h, w)
        """

        batch_size, c, h, w = x.size()
        padded_x = F.pad(x, (1, 1, 1, 1), 'replicate')
        neighbor = F.unfold(padded_x, kernel_size=self.R, dilation=1, stride=1)  # BS, C*R*R, H*W
        neighbor = neighbor.contiguous().view(batch_size, c, self.R, self.R, h, w)
        neighbor = neighbor.permute(0, 2, 3, 1, 4, 5)  # BS, R, R, c, h ,w
        neighbor = neighbor.reshape(batch_size * self.R * self.R, c, h, w)

        theta = self.conv_theta(x)      # BS, C', h, w           # Q
        phi = self.conv_phi(neighbor)   # BS*R*R, C', h, w       # K
        g = self.conv_g(neighbor)       # BS*R*R, C', h, w       # V

        phi = phi.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)                           # K
        phi = phi.permute(0, 4, 5, 3, 1, 2)  # BS, h, w, c, R, R                                            # K
        theta = theta.permute(0, 2, 3, 1).contiguous().view(batch_size, h, w, self.c_out)   # BS, h, w, c   # Q
        theta_dim = theta                                                                                   # Q

        cos_sim = self.CosSimLayer(phi, theta_dim[:, :, :, :, None, None])  # BS, h, w, c, R, R

        softmax_sim = F.softmax(cos_sim.contiguous().view(batch_size, h, w, -1), dim=3).contiguous().view_as(cos_sim)  # BS, h, w, R, R
        softmax_sim = self.att_drop(softmax_sim)

        g = g.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        g = g.permute(0, 4, 5, 1, 2, 3)  # BS, h, w, R, R, c_out

        weighted_g = g * softmax_sim[:, :, :, :, :, None]
        weighted_average = torch.sum(weighted_g.contiguous().view(batch_size, h, w, -1, self.c_out), dim=3)
        weight_average = weighted_average.permute(0, 3, 1, 2).contiguous()  # BS, c_out, h, w

        x_res = self.conv_back(weight_average)
        x_res = self.proj_drop(x_res)

        ret = x + x_res

        return ret