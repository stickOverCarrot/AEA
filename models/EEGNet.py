from abc import ABC

from models.networks import *
import torch


def normalize_adj(adj, mode='sym'):
    assert len(adj.shape) in [2, 3]
    if mode == "sym":
        inv_sqrt_degree = 1. / (torch.sqrt(adj.abs().sum(dim=-1, keepdim=False)) + EOS)
        if len(adj.shape) == 3:
            return inv_sqrt_degree[:, :, None] * adj * inv_sqrt_degree[:, None, :]
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1. / (adj.abs().sum(dim=-1, keepdim=False) + EOS)
        if len(adj.shape) == 3:
            return inv_degree[:, :, None] * adj
        return inv_degree[:, None] * adj
    else:
        exit("wrong norm mode")

class EEGNet(BaseModel):
    def __init__(self,
                 in_chans,
                 input_time_length,
                 n_classes,
                 pool_mode='mean',
                 f1=8,
                 d=2,
                 f2=16,
                 kernel_length=64,
                 drop_prob=0.25,
                 classifier=True,
                 ):
        super(EEGNet, self).__init__()

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self

        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(in_channels=1, out_channels=self.f1,
                      kernel_size=(1, self.kernel_length),
                      stride=1,
                      bias=False,
                      padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3)
        )

        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.f1, self.f1 * self.d, (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1, padding=(0, 0)),
            # nn.Conv2d(self.f1, self.f1 * self.d, (self.in_chans, 1), stride=1, bias=False,
            #           groups=self.f1, padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d, momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.f1 * self.d, self.f1 * self.d, (1, 16), stride=1,
                      bias=False, groups=self.f1 * self.d,
                      padding=(0, 16 // 2)),
            nn.Conv2d(self.f1 * self.d, self.f2, (1, 1), stride=1, bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        if self.classifier:
            out = np_to_var(
                np.ones((1, self.in_chans, self.input_time_length, 1),
                        dtype=np.float32))
            out = self.forward_init(out)
            # out = self.separable_conv(self.spatial_conv(self.temporal_conv(out)))
            n_out_virtual_chans = out.cpu().data.numpy().shape[2]
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time
            self.cls = nn.Sequential(
                nn.Dropout(p=self.drop_prob),
                Conv2dWithConstraint(self.f2, self.n_classes,
                                     (n_out_virtual_chans, self.final_conv_length), max_norm=0.5,
                                     bias=True),
                # nn.Conv2d(self.f2, self.n_classes,
                #          (n_out_virtual_chans, self.final_conv_length), bias=True),
                Expression(_transpose_1_0),
                Expression(_squeeze_final_output),
            )

        self.apply(glorot_weight_zero_bias)

    def forward_init(self, x):
        with th.no_grad():
            for module in self._modules:
                if isinstance(self._modules[module], th.nn.ModuleList):
                    x = self._modules[module][0](x)
                else:
                    x = self._modules[module](x)
        return x

    def forward(self, x):
        bs = x.size(0)//2
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        feats = self.separable_conv(x)
        return feats, self.cls(feats)

    def get_feature_dim(self):
        return self.f2*self.final_conv_length

class EEGNetP(BaseModel):
    def __init__(self,
                 in_chans,
                 input_time_length,
                 n_classes,
                 pool_mode='mean',
                 f1=8,
                 d=2,
                 f2=16,
                 kernel_length=64,
                 drop_prob=0.25,
                 classifier=True,
                 Adj=None,
                 ):
        super(EEGNetP, self).__init__()

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self
        self.input_channel = 1
        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.xs, self.ys = th.tril_indices(self.in_chans, self.in_chans, offset=-1)
        self.diag_idx = th.arange(self.in_chans)
        self.weight_tril = nn.Parameter(th.zeros((self.in_chans, self.in_chans), dtype=th.float32)[self.xs, self.ys],
                                        requires_grad=True)
        # self.weight_tril = nn.Parameter(((th.rand((self.in_chans, self.in_chans), dtype=th.float32)-0.5)*0.2)[self.xs, self.ys],
        #                                 requires_grad=True)
        self.weight_diag = nn.Parameter(1 * th.ones(self.in_chans, dtype=th.float32), requires_grad=True)
        # self.weight_diag = nn.Parameter(th.rand(self.in_chans, dtype=th.float32), requires_grad=True)

        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.f1,
                      kernel_size=(1, self.kernel_length),
                      stride=1,
                      bias=False,
                      padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3)
        )

        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.f1, self.f1 * self.d, (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1, padding=(0, 0)),
            # nn.Conv2d(self.f1, self.f1 * self.d, (self.in_chans, 1), stride=1, bias=False,
            #           groups=self.f1, padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d, momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.f1 * self.d, self.f1 * self.d, (1, 16), stride=1,
                      bias=False, groups=self.f1 * self.d,
                      padding=(0, 16 // 2)),
            nn.Conv2d(self.f1 * self.d, self.f2, (1, 1), stride=1, bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        if self.classifier:
            out = np_to_var(
                np.ones((1, self.in_chans, self.input_time_length, self.input_channel),
                        dtype=np.float32))
            out = self.forward_init(out)
            # out = self.separable_conv(self.spatial_conv(self.temporal_conv(out)))
            n_out_virtual_chans = out.cpu().data.numpy().shape[2]
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time
            self.cls = nn.Sequential(
                nn.Dropout(p=self.drop_prob),
                Conv2dWithConstraint(self.f2, self.n_classes,
                                     (n_out_virtual_chans, self.final_conv_length), max_norm=0.5,
                                     bias=True),
                # nn.Conv2d(self.f2, self.n_classes,
                #          (n_out_virtual_chans, self.final_conv_length), bias=True),
                Expression(_transpose_1_0),
                Expression(_squeeze_final_output),
            )

        self.apply(glorot_weight_zero_bias)


    def forward_init(self, x):
        with th.no_grad():
            for module in self._modules:
                if isinstance(self._modules[module], th.nn.ModuleList):
                    x = self._modules[module][0](x)
                else:
                    x = self._modules[module](x)
        return x

    def forward(self, x, t=True):
        if t:
            edge_weight = th.zeros([self.in_chans, self.in_chans], device=x.device)
            edge_weight[self.xs.to(x.device), self.ys.to(x.device)] = self.weight_tril.to(x.device)
            weight = edge_weight + edge_weight.T
            weight = weight + th.diag(self.weight_diag.to(x.device))
            # weight = normalize_adj(weight)
            # weight*x*weight
            x = th.matmul(weight.unsqueeze(0), x)
            # x2 = th.matmul(x1, weight.unsqueeze(0))
            # x = th.cat([x1, x2], dim=1)
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        feats = self.separable_conv(x)
        # if not self.classifier:
        #     return feats
        return feats, self.cls(feats)

    def get_feature_dim(self):
        return self.f2*self.final_conv_length


def set_cls(drop_prob, f2, n_classes, n_out_virtual_chans=1, final_conv_length=17):
    n = nn.Sequential(
        nn.Dropout(p=drop_prob),
        Conv2dWithConstraint(f2, n_classes,
                             (n_out_virtual_chans, final_conv_length), max_norm=0.5,
                             bias=True),
        Expression(_transpose_1_0),
        Expression(_squeeze_final_output),
    )
    for name, param in n.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, mean=0, std=0.01)  # 正态初始化
        if 'bias' in name:
            nn.init.constant_(param, val=0)
    return n


class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, x):
        return ReverseLayer.apply(x, self.alpha)


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _review(x):
    return x.contiguous().view(-1, x.size(2), x.size(3))


def _flatten(x):
    return x.contiguous().view(x.size(0), -1)


def _squeeze_final_output(x):
    """
    Remove empty dim at end and potentially remove empty time dim
    Do not just use squeeze as we never want to remove first dim
    :param x:
    :return:
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x
