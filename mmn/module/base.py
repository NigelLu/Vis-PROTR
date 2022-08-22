
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.nn.functional import interpolate as resize

# ==> Spatial Context Encoder

def generate_spatial_descriptor(data, kernel_size):
    '''
    Applies self local similarity with fixed sliding window.
    Args:
        data: featuren map, variable of shape (b,c,h,w)
        kernel_size: width/heigh of local window, int
    Returns:
        output: global spatial map, variable of shape (b,c,h,w)
    '''

    padding = int(kernel_size // 2)  # 5.7//2 = 2.0, 5.2//2 = 2.0
    b, c, h, w = data.shape
    p2d = _quadruple(padding)  # (pad_l,pad_r,pad_t,pad_b)
    data_padded = F.pad(data, p2d, 'constant', 0)  # output variable
    assert data_padded.shape == (
    b, c, (h + 2 * padding), (w + 2 * padding)), 'Error: data_padded shape{} wrong!'.format(data_padded.shape)

    output = Variable(torch.zeros(b, kernel_size * kernel_size, h, w), requires_grad=data.requires_grad)
    if data.is_cuda:
        output = output.cuda(data.get_device())

    for hi in range(h):
        for wj in range(w):
            q = data[:, :, hi, wj].contiguous()  # (b,c)
            i = hi + padding  # h index in datapadded
            j = wj + padding  # w index in datapadded

            hs = i - padding
            he = i + padding + 1
            ws = j - padding
            we = j + padding + 1
            patch = data_padded[:, :, hs:he, ws:we].contiguous()  # (b,c,k,k)
            assert (patch.shape == (b, c, kernel_size, kernel_size))
            hk, wk = kernel_size, kernel_size

            # reshape features for matrix multiplication
            feature_a = q.view(b, c, 1 * 1).transpose(1, 2)  # (b,1,c) input is not contigous
            feature_b = patch.view(b, c, hk * wk)  # (b,c,L)  L=k*k

            # perform matrix mult.
            feature_mul = torch.bmm(feature_a, feature_b)  # (b,1,L)
            assert (feature_mul.shape == (b, 1, hk * wk))
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.unsqueeze(1)  # (b,L)
            output[:, :, hi, wj] = correlation_tensor

    return output

def featureL2Norm(feature):  # [B, c ,h, w]
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)

class SpatialContextEncoder(torch.nn.Module):
    '''
    Spatial Context Encoder.
    Author: Shuaiyi Huang
    Input:
        x: feature of shape (b,c,h,w)
    Output:
        feature_embd: context-aware semantic feature of shape (b,c+k**2,h,w), where k is the kernel size of spatial descriptor
    '''

    def __init__(self, kernel_size=None, input_dim=None, hidden_dim=None):
        super(SpatialContextEncoder, self).__init__()
        self.embeddingFea = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.embeddingFea.cuda()
        self.kernel_size = kernel_size
        print('SpatialContextEncoder initialization: input_dim {},hidden_dim {}'.format(input_dim, hidden_dim))

        return

    def forward(self, x):
        kernel_size = self.kernel_size
        feature_gs = generate_spatial_descriptor(x, kernel_size=kernel_size)   # [B, c, h, w]

        # Add L2norm
        feature_gs = featureL2Norm(feature_gs)  # L2 Norm

        # concatenate
        feature_cat = torch.cat([x, feature_gs], 1)    # x 和 feature_gs 都已经经过了 L2 norm

        # embed
        feature_embd = self.embeddingFea(feature_cat)

        return feature_embd

# ==> Provides functions that manipulate boxes and points

class Geometry(object):

    @classmethod
    def initialize(cls, img_size):
        cls.img_size = img_size

        cls.spatial_side = int(img_size / 8)
        cls.feat_idx = torch.arange(0, cls.spatial_side).float()
        norm_grid1d = torch.linspace(-1, 1, cls.spatial_side)
        if torch.cuda.is_available():
            cls.feat_idx = cls.feat_idx.cuda()
            norm_grid1d = norm_grid1d.cuda()

        cls.norm_grid_x = norm_grid1d.view(1, -1).repeat(cls.spatial_side, 1).view(1, 1, -1)
        cls.norm_grid_y = norm_grid1d.view(-1, 1).repeat(1, cls.spatial_side).view(1, 1, -1)
        cls.grid = torch.stack(list(reversed(torch.meshgrid(norm_grid1d, norm_grid1d)))).permute(1, 2, 0)

    @classmethod
    def normalize_kps(cls, kps):
        kps = kps.clone().detach()
        kps[kps != -2] -= (cls.img_size // 2)
        kps[kps != -2] /= (cls.img_size // 2)
        return kps

    @classmethod
    def unnormalize_kps(cls, kps):
        kps = kps.clone().detach()
        kps[kps != -2] *= (cls.img_size // 2)
        kps[kps != -2] += (cls.img_size // 2)
        return kps

    @classmethod
    def attentive_indexing(cls, kps, thres=0.1):
        r"""kps: normalized keypoints x, y (N, 2)
            returns attentive index map(N, spatial_side, spatial_side)
        """
        nkps = kps.size(0)
        kps = kps.view(nkps, 1, 1, 2)

        eps = 1e-5
        attmap = (cls.grid.unsqueeze(0).repeat(nkps, 1, 1, 1) - kps).pow(2).sum(dim=3)
        attmap = (attmap + eps).pow(0.5)
        attmap = (thres - attmap).clamp(min=0).view(nkps, -1)
        attmap = attmap / attmap.sum(dim=1, keepdim=True)
        attmap = attmap.view(nkps, cls.spatial_side, cls.spatial_side)

        return attmap

    @classmethod
    def apply_gaussian_kernel(cls, corr, sigma=17):
        bsz, side, side = corr.size()

        center = corr.max(dim=2)[1]
        center_y = center // cls.spatial_side
        center_x = center % cls.spatial_side

        y = cls.feat_idx.view(1, 1, cls.spatial_side).repeat(bsz, center_y.size(1), 1) - center_y.unsqueeze(2)
        x = cls.feat_idx.view(1, 1, cls.spatial_side).repeat(bsz, center_x.size(1), 1) - center_x.unsqueeze(2)

        y = y.unsqueeze(3).repeat(1, 1, 1, cls.spatial_side)
        x = x.unsqueeze(2).repeat(1, 1, cls.spatial_side, 1)

        gauss_kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        filtered_corr = gauss_kernel * corr.view(bsz, -1, cls.spatial_side, cls.spatial_side)
        filtered_corr = filtered_corr.view(bsz, side, side)

        return filtered_corr

    @classmethod
    def transfer_kps(cls, confidence_ts, src_kps, n_pts, normalized):
        r""" Transfer keypoints by weighted average """

        if not normalized:
            src_kps = Geometry.normalize_kps(src_kps)
        confidence_ts = cls.apply_gaussian_kernel(confidence_ts)

        pdf = F.softmax(confidence_ts, dim=2)
        prd_x = (pdf * cls.norm_grid_x).sum(dim=2)
        prd_y = (pdf * cls.norm_grid_y).sum(dim=2)

        prd_kps = []
        for idx, (x, y, src_kp, np) in enumerate(zip(prd_x, prd_y, src_kps, n_pts)):
            max_pts = src_kp.size()[1]
            prd_xy = torch.stack([x, y]).t()

            src_kp = src_kp[:, :np].t()
            attmap = cls.attentive_indexing(src_kp).view(np, -1)
            prd_kp = (prd_xy.unsqueeze(0) * attmap.unsqueeze(-1)).sum(dim=1).t()
            pads = (torch.zeros((2, max_pts - np)) - 2)                                  # check
            if torch.cuda.is_available():
                pads = pads.cuda()
            prd_kp = torch.cat([prd_kp, pads], dim=1)
            prd_kps.append(prd_kp)

        return torch.stack(prd_kps)

    @staticmethod
    def get_coord1d(coord4d, ksz):
        i, j, k, l = coord4d
        coord1d = i * (ksz ** 3) + j * (ksz ** 2) + k * (ksz) + l
        return coord1d

    @staticmethod
    def get_distance(coord1, coord2):
        delta_y = int(math.pow(coord1[0] - coord2[0], 2))
        delta_x = int(math.pow(coord1[1] - coord2[1], 2))
        dist = delta_y + delta_x
        return dist

    @staticmethod
    def interpolate4d(tensor4d, size):
        bsz, h1, w1, h2, w2 = tensor4d.size()
        tensor4d = tensor4d.view(bsz, h1, w1, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, size, mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, h2, w2, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, size, mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, size[0], size[0], size[0], size[0])

        return tensor4d
    @staticmethod
    def init_idx4d(ksz):
        i0 = torch.arange(0, ksz).repeat(ksz ** 3)  # [625]
        i1 = torch.arange(0, ksz).unsqueeze(1).repeat(1, ksz).view(-1).repeat(ksz ** 2)
        i2 = torch.arange(0, ksz).unsqueeze(1).repeat(1, ksz ** 2).view(-1).repeat(ksz)
        i3 = torch.arange(0, ksz).unsqueeze(1).repeat(1, ksz ** 3).view(-1)
        idx4d = torch.stack([i3, i2, i1, i0]).t().numpy()

        return idx4d

# ==> Provides functions that creates/manipulates correlation matrices

class Correlation:

    @classmethod
    def mutual_nn_filter(cls, correlation_matrix, eps=1e-30):
        r""" Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18 )"""
        corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += eps
        corr_trg_max[corr_trg_max == 0] += eps

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)

    @classmethod
    def build_correlation6d(self, src_feat, trg_feat, scales, conv2ds):
        r""" Build 6-dimensional correlation tensor """

        bsz, _, side, side = src_feat.size()

        # Construct feature pairs with multiple scales
        _src_feats = []
        _trg_feats = []
        for scale, conv in zip(scales, conv2ds):
            s = (round(side * math.sqrt(scale)),) * 2
            _src_feat = conv(resize(src_feat, s, mode='bilinear', align_corners=True))
            _trg_feat = conv(resize(trg_feat, s, mode='bilinear', align_corners=True))
            _src_feats.append(_src_feat)
            _trg_feats.append(_trg_feat)

        # Build multiple 4-dimensional correlation tensor
        corr6d = []
        for src_feat in _src_feats:
            ch = src_feat.size(1)

            src_side = src_feat.size(-1)
            src_feat = src_feat.view(bsz, ch, -1).transpose(1, 2)   # [B, n_q, ch]
            src_norm = src_feat.norm(p=2, dim=2, keepdim=True)      # [B, n, 1]

            for trg_feat in _trg_feats:
                trg_side = trg_feat.size(-1)
                trg_feat = trg_feat.view(bsz, ch, -1)               # [B, ch, n_s]
                trg_norm = trg_feat.norm(p=2, dim=1, keepdim=True)  # [B, 1, n]

                correlation = torch.bmm(src_feat, trg_feat) / torch.bmm(src_norm, trg_norm)
                correlation = correlation.view(bsz, src_side, src_side, trg_side, trg_side).contiguous()   # [B,h,w,h,w]
                corr6d.append(correlation)

        # Resize the spatial sizes of the 4D tensors to the same size
        for idx, correlation in enumerate(corr6d):
            corr6d[idx] = Geometry.interpolate4d(correlation, [side, side])

        # Build 6-dimensional correlation tensor
        corr6d = torch.stack(corr6d).view(len(scales), len(scales),
                                          bsz, side, side, side, side).permute(2, 0, 1, 3, 4, 5, 6)
        return corr6d.clamp(min=0)    # min bound 0

# ==> CHM 4D kernel (psi, iso, and full) generator

class KernelGenerator:
    def __init__(self, ksz, ktype):
        self.ksz = ksz
        self.idx4d = Geometry.init_idx4d(ksz)
        self.kernel = torch.zeros((ksz, ksz, ksz, ksz))
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.center = (ksz // 2, ksz // 2)
        self.ktype = ktype

    def quadrant(self, crd):
        if crd[0] < self.center[0]:
            horz_quad = -1
        elif crd[0] < self.center[0]:
            horz_quad = 1
        else:
            horz_quad = 0

        if crd[1] < self.center[1]:
            vert_quad = -1
        elif crd[1] < self.center[1]:
            vert_quad = 1
        else:
            vert_quad = 0

        return horz_quad, vert_quad

    def generate(self):
        return None if self.ktype == 'full' else self.generate_chm_kernel()

    def generate_chm_kernel(self):
        param_dict = {}
        for idx in self.idx4d:  # [625, 4], each row is a combination of src_ij and trg_ij
            src_i, src_j, trg_i, trg_j = idx
            d_tail = Geometry.get_distance((src_i, src_j), self.center)
            d_head = Geometry.get_distance((trg_i, trg_j), self.center)
            d_off = Geometry.get_distance((src_i, src_j), (trg_i, trg_j))
            horz_quad, vert_quad = self.quadrant((src_j, src_i))

            src_crd = (src_i, src_j)
            trg_crd = (trg_i, trg_j)

            key = self.build_key(horz_quad, vert_quad, d_head, d_tail, src_crd, trg_crd, d_off)  # 'psi': [d_max, d_min, d_off]
            coord1d = Geometry.get_coord1d((src_i, src_j, trg_i, trg_j), self.ksz)               # it's corresponding idx in self.idx4d

            if param_dict.get(key) is None: param_dict[key] = []
            param_dict[key].append(coord1d)

        return param_dict

    def build_key(self, horz_quad, vert_quad, d_head, d_tail, src_crd, trg_crd, d_off):

        if self.ktype == 'iso':
            return '%d' % d_off
        elif self.ktype == 'psi':
            d_max = max(d_head, d_tail)
            d_min = min(d_head, d_tail)
            return '%d_%d_%d' % (d_max, d_min, d_off)
        else:
            raise Exception('not implemented.')

# ==> 4D and 6D convolutional Hough matching layers

def fast4d(corr, kernel, bias=None):
    r""" Optimized implementation of 4D convolution """
    bsz, ch, srch, srcw, trgh, trgw = corr.size()
    out_channels, _, kernel_size, kernel_size, kernel_size, kernel_size = kernel.size()
    psz = kernel_size // 2

    out_corr = torch.zeros((bsz, out_channels, srch, srcw, trgh, trgw))
    if torch.cuda.is_available():
        out_corr = out_corr.cuda()
    corr = corr.transpose(1, 2).contiguous().view(bsz * srch, ch, srcw, trgh, trgw)

    for pidx, k3d in enumerate(kernel.permute(2, 0, 1, 3, 4, 5)):  # kernel: [ch, 1, 5, 5, 5, 5]
        inter_corr = F.conv3d(corr, k3d, bias=None, stride=1, padding=psz)
        inter_corr = inter_corr.view(bsz, srch, out_channels, srcw, trgh, trgw).transpose(1, 2).contiguous()

        add_sid = max(psz - pidx, 0)
        add_fid = min(srch, srch + psz - pidx)
        slc_sid = max(pidx - psz, 0)
        slc_fid = min(srch, srch - psz + pidx)

        out_corr[:, :, add_sid:add_fid, :, :, :] += inter_corr[:, :, slc_sid:slc_fid, :, :, :]

    if bias is not None:
        out_corr += bias.view(1, out_channels, 1, 1, 1, 1)

    return out_corr

def fast6d(corr, kernel, bias, diagonal_idx):
    r""" Optimized implementation of 6D convolutional Hough matching
         NOTE: this function only supports kernel size of (3, 3, 5, 5, 5, 5).
    r"""
    bsz, _, s6d, s6d, s4d, s4d, s4d, s4d = corr.size()
    _, _, ks6d, ks6d, ks4d, ks4d, ks4d, ks4d = kernel.size()
    corr = corr.permute(0, 2, 3, 1, 4, 5, 6, 7).contiguous().view(-1, 1, s4d, s4d, s4d, s4d)
    kernel = kernel.view(-1, ks6d ** 2, ks4d, ks4d, ks4d, ks4d).transpose(0, 1)
    corr = fast4d(corr, kernel).view(bsz, s6d * s6d, ks6d * ks6d, s4d, s4d, s4d, s4d)
    corr = corr.view(bsz, s6d, s6d, ks6d, ks6d, s4d, s4d, s4d, s4d).transpose(2, 3).\
        contiguous().view(-1, s6d * ks6d, s4d, s4d, s4d, s4d)

    ndiag = s6d + (ks6d // 2) * 2
    first_sum = []
    for didx in diagonal_idx:
        first_sum.append(corr[:, didx, :, :, :, :].sum(dim=1))
    first_sum = torch.stack(first_sum).transpose(0, 1).view(bsz, s6d * ks6d, ndiag, s4d, s4d, s4d, s4d)

    corr = []
    for didx in diagonal_idx:
        corr.append(first_sum[:, didx, :, :, :, :, :].sum(dim=1))
    sidx = ks6d // 2
    eidx = ndiag - sidx
    corr = torch.stack(corr).transpose(0, 1)[:, sidx:eidx, sidx:eidx, :, :, :, :].unsqueeze(1).contiguous()
    corr += bias.view(1, -1, 1, 1, 1, 1, 1, 1)

    reverse_idx = torch.linspace(s6d * s6d - 1, 0, s6d * s6d).long()
    if torch.cuda.is_available():
        reverse_idx = reverse_idx.cuda()
    corr = corr.view(bsz, 1, s6d * s6d, s4d, s4d, s4d, s4d)[:, :, reverse_idx, :, :, :, :].\
        view(bsz, 1, s6d, s6d, s4d, s4d, s4d, s4d)
    return corr

def init_param_idx4d(param_dict):
    param_idx = []
    for key in param_dict:
        curr_offset = int(key.split('_')[-1])
        idx = torch.tensor(param_dict[key]).cuda() if torch.cuda.is_available() else torch.tensor(param_dict[key])
        param_idx.append(idx)
    return param_idx

class CHM4d(_ConvNd):
    r""" 4D convolutional Hough matching layer
         NOTE: this function only supports in_channels=1 and out_channels=1.
    r"""
    def __init__(self, in_channels, out_channels, ksz4d, ktype, bias=True):
        super(CHM4d, self).__init__(in_channels, out_channels, (ksz4d,) * 4,
                                    (1,) * 4, (0,) * 4, (1,) * 4, False, (0,) * 4,
                                    1, bias, padding_mode='zeros')

        # Zero kernel initialization
        self.zero_kernel4d = torch.zeros((in_channels, out_channels, ksz4d, ksz4d, ksz4d, ksz4d))
        if torch.cuda.is_available():
            self.zero_kernel4d = self.zero_kernel4d.cuda()
        self.nkernels = in_channels * out_channels

        # Initialize kernel indices
        param_dict4d = chm_kernel.KernelGenerator(ksz4d, ktype).generate()
        param_shared =  param_dict4d is not None

        if param_shared:
            # Initialize the shared parameters (multiplied by the number of times being shared)
            self.param_idx = init_param_idx4d(param_dict4d)   # list of list of idx_1d of the 'param4d params sharing wt', 'psi': 55 learnable params
            weights = torch.abs(torch.randn(len(self.param_idx) * self.nkernels)) * 1e-3  # [55]
            for i in range(len(weights)):
                weights[i] = weights[i] * len( self.param_idx[i] )
            # for weight, param_idx in zip(weights.sort()[0], self.param_idx):    weight *= len(param_idx)    # modify the weight in place?

            self.weight = nn.Parameter(weights)
        else:  # full kernel initialziation
            self.param_idx = None
            self.weight = nn.Parameter(torch.abs(self.weight))
            if bias: self.bias = nn.Parameter(torch.tensor(0.0))
        print('(%s) # params in CHM 4D: %d' % (ktype, len(self.weight.view(-1))))

    def forward(self, x):
        kernel = self.init_kernel()
        x = fast4d(x, kernel, self.bias)
        return x

    def init_kernel(self):
        # Initialize CHM kernel (divided by the number of times being shared)
        ksz = self.kernel_size[-1]
        if self.param_idx is None:
            kernel = self.weight
        else:
            kernel = torch.zeros_like(self.zero_kernel4d).view(-1, ksz**4)  # [1, 1, 5, 5, 5, 5] -> [1, 5, 5, 5, 5] 为了支持有多个kernel的情况
            for idx, pdx in enumerate(self.param_idx):     # list of list (sublist 为 row_idx of i, j, k, l share weight)
                for jdx in range(len(kernel)):
                    weight = self.weight[idx + jdx * len(self.param_idx)].repeat(len(pdx)) / len(pdx)
                    kernel[jdx, pdx] = kernel[jdx, pdx] + weight
            kernel = kernel.view(self.in_channels, self.out_channels, ksz, ksz, ksz, ksz)
        return kernel

class CHM6d(_ConvNd):
    r""" 6D convolutional Hough matching layer with kernel (3, 3, 5, 5, 5, 5)
         NOTE: this function only supports in_channels=1 and out_channels=1.
    r"""
    def __init__(self, in_channels, out_channels, ksz6d, ksz4d, ktype):
        kernel_size = (ksz6d, ksz6d, ksz4d, ksz4d, ksz4d, ksz4d)
        super(CHM6d, self).__init__(in_channels, out_channels, kernel_size, (1,) * 6,
                                    (0,) * 6, (1,) * 6, False, (0,) * 6,
                                    1, bias=True, padding_mode='zeros')

        # Zero kernel initialization
        self.zero_kernel4d = torch.zeros((ksz4d, ksz4d, ksz4d, ksz4d))
        self.zero_kernel6d = torch.zeros((ksz6d, ksz6d, ksz4d, ksz4d, ksz4d, ksz4d))
        if torch.cuda.is_available():
            self.zero_kernel4d = self.zero_kernel4d.cuda()
            self.zero_kernel6d = self.zero_kernel6d.cuda()

        self.nkernels = in_channels * out_channels

        # Initialize kernel indices
        # Indices in scale-space where 4D convolutions are performed (3 by 3 scale-space)
        if torch.cuda.is_available():
            self.diagonal_idx = [torch.tensor(x).cuda() for x in [[6], [3, 7], [0, 4, 8], [1, 5], [2]]]
        else:
            self.diagonal_idx = [torch.tensor(x) for x in [[6], [3, 7], [0, 4, 8], [1, 5], [2]]]
        param_dict4d = chm_kernel.KernelGenerator(ksz4d, ktype).generate()
        param_shared =  param_dict4d is not None

        if param_shared:  # psi & iso kernel initialization
            if ktype == 'psi':
                self.param_dict6d = [[4], [0, 8], [2, 6], [1, 3, 5, 7]]
            elif ktype == 'iso':
                self.param_dict6d = [[0, 4, 8], [2, 6], [1, 3, 5, 7]]

            if torch.cuda.is_available():
                self.param_dict6d = [torch.tensor(i).cuda() for i in self.param_dict6d]
            else:
                self.param_dict6d = [torch.tensor(i) for i in self.param_dict6d]

            # Initialize the shared parameters (multiplied by the number of times being shared)
            self.param_idx = init_param_idx4d(param_dict4d)     # list of list for idx1d of the (4d kernel weight sharing)
            self.param = []
            for param_dict6d in self.param_dict6d:
                weights = torch.abs(torch.randn(len(self.param_idx))) * 1e-3    # each cross scale combination's conv4d params [55]
                for i in range(len(weights)):
                    weights[i] = weights[i] * (len(self.param_idx[i]) * len(param_dict6d))  # 为什么要先乘再除
                self.param.append(nn.Parameter(weights))        # ordered by param_dict6d, param_dict4d    # total 55*
            self.param = nn.ParameterList(self.param)    # size:[sn, ln], row idx: scale_offset(param_dict6d) column idx
        else:  # full kernel initialziation
            self.param_idx = None
            self.param = nn.Parameter(torch.abs(self.weight) * 1e-3)
        print('(%s) # params in CHM 6D: %d' % (ktype, sum([len(x.view(-1)) for x in self.param])))
        self.weight = None

    def forward(self, corr):
        kernel = self.init_kernel()
        corr = fast6d(corr, kernel, self.bias, self.diagonal_idx)
        return corr

    def init_kernel(self):
        # Initialize CHM kernel (divided by the number of times being shared)
        if self.param_idx is None:
            return self.param

        kernel6d = torch.zeros_like(self.zero_kernel6d)
        for idx, (param, param_dict6d) in enumerate(zip(self.param, self.param_dict6d)):
            ksz4d = self.kernel_size[-1]   # 此处param只 针对当前scale pair (所有share 参数的scale pair)
            kernel4d = torch.zeros_like(self.zero_kernel4d).view(-1)   # [5, 5, 5, 5] -> [525]
            for jdx, pdx in enumerate(self.param_idx):  # list of list (sublist 为 row_idx of i, j, k, l share weight)
                kernel4d[pdx] = ((param[jdx] / len(pdx)) / len(param_dict6d))
            kernel6d.view(-1, ksz4d, ksz4d, ksz4d, ksz4d)[param_dict6d] = kernel4d.view(ksz4d, ksz4d, ksz4d, ksz4d)   # [9, 5, 5, 5]
        kernel6d = kernel6d.unsqueeze(0).unsqueeze(0)

        return kernel6d
