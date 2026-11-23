# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from ultralytics.nn.modules.conv import Conv
# from .block import C2f

    
# class DWT(nn.Module):
#     def __init__(self): 
#         super().__init__()

#     def forward(self, x):
#         x01 = x[:, :, 0::2, :] / 2
#         x02 = x[:, :, 1::2, :] / 2
#         x1 = x01[:, :, :, 0::2]
#         x2 = x02[:, :, :, 0::2]
#         x3 = x01[:, :, :, 1::2]
#         x4 = x02[:, :, :, 1::2]
#         x_LL = x1 + x2 + x3 + x4
#         x_HL = -x1 - x2 + x3 + x4
#         x_LH = -x1 + x2 - x3 + x4
#         x_HH = x1 - x2 - x3 + x4
#         return x_LL, x_LH, x_HL, x_HH

# class IDWT(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x_LL, x_LH, x_HL, x_HH):
#         x1 = (x_LL - x_LH - x_HL + x_HH) / 2.0
#         x2 = (x_LL + x_LH - x_HL - x_HH) / 2.0
#         x3 = (x_LL - x_LH + x_HL - x_HH) / 2.0
#         x4 = (x_LL + x_LH + x_HL + x_HH) / 2.0
#         B, C, H_half, W_half = x_LL.shape
#         top = torch.stack([x1, x3], dim=-1).view(B, C, H_half, W_half * 2)
#         bottom = torch.stack([x2, x4], dim=-1).view(B, C, H_half, W_half * 2)
#         x = torch.stack([top, bottom], dim=-2).view(B, C, H_half * 2, W_half * 2)
#         return x

# def normal_init(module, mean=0, std=1, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.normal_(module.weight, mean, std)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)


# def constant_init(module, val, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.constant_(module.weight, val)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)

# class DySample(nn.Module):
#     def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True):
#         super().__init__()
#         self.scale = scale
#         self.style = style
#         self.groups = groups
#         assert style in ['lp', 'pl']
#         if style == 'pl':
#             assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
#         assert in_channels >= groups and in_channels % groups == 0

#         if style == 'pl':
#             in_channels = in_channels // scale ** 2
#             out_channels = 2 * groups
#         else:
#             out_channels = 2 * groups * scale ** 2

#         self.offset = nn.Conv2d(in_channels, out_channels, 1)
#         normal_init(self.offset, std=0.001)
#         if dyscope:
#             self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#             constant_init(self.scope, val=0.)

#         self.register_buffer('init_pos', self._init_pos())

#     def _init_pos(self):
#         h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
#         return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

#     def sample(self, x, offset):
#         B, _, H, W = offset.shape
#         offset = offset.view(B, 2, -1, H, W)
#         coords_h = torch.arange(H) + 0.5
#         coords_w = torch.arange(W) + 0.5
#         coords = torch.stack(torch.meshgrid([coords_w, coords_h])
#                              ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
#         normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
#         coords = 2 * (coords + offset) / normalizer - 1
#         coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
#             B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
#         return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
#                              align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

#     def forward_lp(self, x):
#         if hasattr(self, 'scope'):
#             offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
#         else:
#             offset = self.offset(x) * 0.25 + self.init_pos
#         return self.sample(x, offset)

#     def forward_pl(self, x):
#         x_ = F.pixel_shuffle(x, self.scale)
#         if hasattr(self, 'scope'):
#             offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
#         else:
#             offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
#         return self.sample(x, offset)

#     def forward(self, x):
#         if self.style == 'pl':
#             return self.forward_pl(x)
#         return self.forward_lp(x)



# # class Attention(nn.Module):
# #     def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
# #         super(Attention, self).__init__()
# #         attention_channel = max(int(in_planes * reduction), min_channel)
# #         self.kernel_size = kernel_size
# #         self.kernel_num = kernel_num
# #         self.temperature = 1.0

# #         self.avgpool = nn.AdaptiveAvgPool2d(1)
# #         self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
# #         self.bn = nn.BatchNorm2d(attention_channel)
# #         self.relu = nn.ReLU(inplace=True)

# #         self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
# #         self.func_channel = self.get_channel_attention

# #         if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
# #             self.func_filter = self.skip
# #         else:
# #             self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
# #             self.func_filter = self.get_filter_attention

# #         if kernel_size == 1:  # point-wise convolution
# #             self.func_spatial = self.skip
# #         else:
# #             self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
# #             self.func_spatial = self.get_spatial_attention

# #         if kernel_num == 1:
# #             self.func_kernel = self.skip
# #         else:
# #             self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
# #             self.func_kernel = self.get_kernel_attention

# #         self._initialize_weights()

# #     def _initialize_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# #                 if m.bias is not None:
# #                     nn.init.constant_(m.bias, 0)
# #             if isinstance(m, nn.BatchNorm2d):
# #                 nn.init.constant_(m.weight, 1)
# #                 nn.init.constant_(m.bias, 0)

# #     def update_temperature(self, temperature):
# #         self.temperature = temperature

# #     @staticmethod
# #     def skip(_):
# #         return 1.0

# #     def get_channel_attention(self, x):
# #         channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
# #         return channel_attention

# #     def get_filter_attention(self, x):
# #         filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
# #         return filter_attention

# #     def get_spatial_attention(self, x):
# #         spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
# #         spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
# #         return spatial_attention

# #     def get_kernel_attention(self, x):
# #         kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
# #         kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
# #         return kernel_attention

# #     def forward(self, x):
# #         x = self.avgpool(x)
# #         x = self.fc(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


# # class ODConv2d(nn.Module):
# #     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
# #                  reduction=0.0625, kernel_num=4):
# #         super(ODConv2d, self).__init__()
# #         self.in_planes = in_planes
# #         self.out_planes = out_planes
# #         self.kernel_size = kernel_size
# #         self.stride = stride
# #         self.padding = padding
# #         self.dilation = dilation
# #         self.groups = groups
# #         self.kernel_num = kernel_num
# #         self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
# #                                    reduction=reduction, kernel_num=kernel_num)
# #         self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
# #                                    requires_grad=True)
# #         self._initialize_weights()

# #         if self.kernel_size == 1 and self.kernel_num == 1:
# #             self._forward_impl = self._forward_impl_pw1x
# #         else:
# #             self._forward_impl = self._forward_impl_common

# #     def _initialize_weights(self):
# #         for i in range(self.kernel_num):
# #             nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

# #     def update_temperature(self, temperature):
# #         self.attention.update_temperature(temperature)

# #     def _forward_impl_common(self, x):
# #         # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
# #         # while we observe that when using the latter method the models will run faster with less gpu memory cost.
# #         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
# #         batch_size, in_planes, height, width = x.size()
# #         x = x * channel_attention
# #         x = x.reshape(1, -1, height, width)
# #         aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
# #         aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
# #             [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
# #         output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
# #                           dilation=self.dilation, groups=self.groups * batch_size)
# #         output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
# #         output = output * filter_attention
# #         return output

# #     def _forward_impl_pw1x(self, x):
# #         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
# #         x = x * channel_attention
# #         output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
# #                           dilation=self.dilation, groups=self.groups)
# #         output = output * filter_attention
# #         return output

# #     def forward(self, x):
# #         return self._forward_impl(x)

# class ConvBN(torch.nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         super().__init__()
#         self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
#         if with_bn:
#             self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
#             torch.nn.init.constant_(self.bn.weight, 1)
#             torch.nn.init.constant_(self.bn.bias, 0)
# class GRN(nn.Module):
#     def __init__(self, channels, eps=1e-6):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
#         self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
#         self.eps = eps

#     def forward(self, x):
#         gx = torch.sqrt((x ** 2).mean(dim=(2, 3), keepdim=True) + self.eps)
#         nx = x / (gx + self.eps)
#         return x + self.gamma * nx + self.beta

# class Attention(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
#         super(Attention, self).__init__()
#         attention_channel = max(int(in_planes * reduction), min_channel)
        

#         self.kernel_h, self.kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.kernel_num = kernel_num
#         self.temperature = 1.0

#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
#         self.bn = nn.BatchNorm2d(attention_channel)
#         self.relu = nn.ReLU(inplace=True)

#         self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
#         self.func_channel = self.get_channel_attention
#         if in_planes <= min_channel:
#             attention_channel = in_planes
#         else:
#             attention_channel = max(int(in_planes * reduction), min_channel)
        
#         if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
#             self.func_filter = self.skip
#         else:
#             self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
#             self.func_filter = self.get_filter_attention

#         if self.kernel_h == 1 and self.kernel_w == 1:  
#             self.func_spatial = self.skip
#         else:
#             self.spatial_fc = nn.Conv2d(attention_channel, self.kernel_h * self.kernel_w, 1, bias=True)
#             self.func_spatial = self.get_spatial_attention

#         if kernel_num == 1:
#             self.func_kernel = self.skip
#         else:
#             self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
#             self.func_kernel = self.get_kernel_attention

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def update_temperature(self, temperature):
#         self.temperature = temperature

#     @staticmethod
#     def skip(_):
#         return 1.0

#     def get_channel_attention(self, x):
#         channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
#         return channel_attention

#     def get_filter_attention(self, x):
#         filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
#         return filter_attention

#     def get_spatial_attention(self, x):
#         # SỬA Ở ĐÂY: Reshape theo kernel_h và kernel_w
#         spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_h, self.kernel_w)
#         spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
#         return spatial_attention

#     def get_kernel_attention(self, x):
#         kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
#         kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
#         return kernel_attention

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

# class ODConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#                  reduction=0.0625, kernel_num=4):
#         super(ODConv2d, self).__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes

#         self.kernel_h, self.kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.kernel_num = kernel_num
#         self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
#                                    reduction=reduction, kernel_num=kernel_num)
                                   

#         self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, self.kernel_h, self.kernel_w),
#                                    requires_grad=True)
#         self._initialize_weights()

#         if self.kernel_h == 1 and self.kernel_w == 1 and self.kernel_num == 1:
#             self._forward_impl = self._forward_impl_pw1x
#         else:
#             self._forward_impl = self._forward_impl_common

#     def _initialize_weights(self):
#         for i in range(self.kernel_num):
#             nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

#     def update_temperature(self, temperature):
#         self.attention.update_temperature(temperature)

#     def _forward_impl_common(self, x):
#         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
#         batch_size, in_planes, height, width = x.size()
#         x = x * channel_attention
#         x = x.reshape(1, -1, height, width)
#         aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)

#         aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
#             [-1, self.in_planes // self.groups, self.kernel_h, self.kernel_w])
            
#         output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                           dilation=self.dilation, groups=self.groups * batch_size)
#         output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
#         output = output * filter_attention
#         return output

#     def _forward_impl_pw1x(self, x):
#         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
#         x = x * channel_attention
#         output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
#                           dilation=self.dilation, groups=self.groups)
#         output = output * filter_attention
#         return output

#     def forward(self, x):
#         return self._forward_impl(x)

# class AFSA(nn.Module):
#     def __init__(self, c, shortcut=True):
#         super().__init__()
#         self.dwt = DWT()
#         self.idwt = IDWT()
#         self.dw3x3 = nn.Conv2d(4 * c, 4 * c, kernel_size=3, stride=1, padding=1, groups = 4 * c, bias=False)
#         self.branch1 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=(5, 1), stride=1, padding=(2, 0)),
#             nn.Conv2d(c, c, kernel_size=(5, 1), padding=(2, 0), groups=c, bias=False),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
#         self.branch2 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=(1, 5), stride=1, padding=(0, 2)),
#             nn.Conv2d(c, c, kernel_size=(1, 5), padding=(0, 2), groups=c, bias=False),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
        
#         self.mask = nn.Sequential(
#             nn.Conv2d(c, c // 2, kernel_size=1, bias=False),
#             nn.BatchNorm2d(c // 2),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(c // 2, c, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
#         self.pwconv = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
#         self.shortcut = shortcut
#     def forward(self, x):
#         LL, LH, HL, HH = self.dwt(x)
#         fcat = self.dw3x3(torch.cat([LL, LH, HL, HH], dim=1))
#         fLL, fLH, fHL, fHH = torch.chunk(fcat, 4, dim=1)
#         out1 = self.idwt(fLL, fLH, fHL, fHH)
#         f_mix = self.pwconv(out1)
#         f1 = self.branch1(f_mix)
#         f2 = self.branch2(f_mix)
#         f3 = f1 + f2 # c channels 
#         f3 = x * self.mask(f3)
        
#         if self.shortcut:
#             return x + f3
#         else:
#             return f3
        
# class C3k2_AFSA(nn.Module):

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e) 
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.m = nn.ModuleList(AFSA(self.c, shortcut) for _ in range(n))
#         self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)

#         return self.cv2(torch.cat(y, 1))

# class AODC(nn.Module):
#     def __init__(self, c, shortcut=True):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             ODConv2d(c, c, kernel_size=3, stride=1, padding=1, kernel_num=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             ODConv2d(c, c, kernel_size=5, stride=1, padding=2, kernel_num=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
        
#         self.branch1 = ODConv2d(c, c, kernel_size=(1, 7), stride=1, padding=(0, 3), kernel_num=1)
#         self.branch2 = ODConv2d(c, c, kernel_size=(7, 1), stride=1, padding=(3, 0), kernel_num=1)
#         self.norm = nn.Sequential(
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
#     def forward(self, x):
#         f1 = self.conv1(x)
#         f2 = self.conv2(f1)
#         branch1 = self.branch1(f2)
#         branch2 = self.branch2(f2)
#         out = branch1 + branch2
#         out = self.norm(out)
#         return x + out
# class PKSModule(nn.Module):

#     def __init__(self, dim, branch_scale=1.0):
#         super().__init__()
#         self.dim = dim
#         self.branch_scale = branch_scale / math.sqrt(5)

#         self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)

#         k_axial = 15
#         self.branch1_axial = nn.Sequential(
#             nn.Conv2d(dim, dim, (1, k_axial), stride=1, padding=(0, k_axial//2), groups=dim, bias=False),
#             nn.Conv2d(dim, dim, (k_axial, 1), stride=1, padding=(k_axial//2, 0), groups=dim, bias=False),
#             nn.BatchNorm2d(dim)
#         )

#         k_b2, d_b2 = 7, 2
#         pad_b2 = (k_b2 - 1) * d_b2 // 2
#         self.branch2_sparse = nn.Sequential(
#             nn.Conv2d(dim, dim, k_b2, stride=1, padding=pad_b2, dilation=d_b2, groups=dim, bias=False),
#             nn.BatchNorm2d(dim)
#         )

#         k_b3, d_b3 = 5, 2
#         pad_b3 = (k_b3 - 1) * d_b3 // 2
#         self.branch3_sparse = nn.Sequential(
#             nn.Conv2d(dim, dim, k_b3, stride=1, padding=pad_b3, dilation=d_b3, groups=dim, bias=False),
#             nn.BatchNorm2d(dim)
#         )

#         k_b4, d_b4 = 3, 3
#         pad_b4 = (k_b4 - 1) * d_b4 // 2
#         self.branch4_sparse = nn.Sequential(
#             nn.Conv2d(dim, dim, k_b4, stride=1, padding=pad_b4, dilation=d_b4, groups=dim, bias=False),
#             nn.BatchNorm2d(dim)
#         )

#         self.branch5_dense = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
#             nn.BatchNorm2d(dim)
#         )

#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

#     def forward(self, x):
#         x_feat = self.conv0(x)

#         attn = self.branch1_axial(x_feat)
#         attn = attn + self.branch2_sparse(x_feat)
#         attn = attn + self.branch3_sparse(x_feat)
#         attn = attn + self.branch4_sparse(x_feat)
#         attn = attn + self.branch5_dense(x_feat)

#         attn = attn * self.branch_scale

#         attn = self.conv1(attn)

#         return x * attn
# class PKSBlock(nn.Module):

#     def __init__(self, c, shortcut=True, branch_scale=1.0):
#         super().__init__()

#         self.proj_1 = ConvBN(c, c, kernel_size=1)
#         self.act = nn.GELU()
        
#         self.pks = PKSModule(c, branch_scale=branch_scale)
        
#         self.proj_2 = ConvBN(c, c, kernel_size=1)
#         self.shortcut = shortcut
#         if self.shortcut:
#             self.layer_scale = nn.Parameter(1e-2 * torch.ones((c)), requires_grad=True)

#     def forward(self, x):
#         out = self.proj_1(x)
#         out = self.act(out)

#         out = self.pks(out)
        
#         out = self.proj_2(out)

#         if self.shortcut:
#             return x + out * self.layer_scale.unsqueeze(-1).unsqueeze(-1)
#         else:
        
#             return out

# class SCR(nn.Module): 
#     def __init__(self, c1, reduction=16):
#         super().__init__()

#         self.pks = PKSBlock(c1)
#         self.grn = GRN(c1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         t = int(abs((math.log2(c1) + 1) / 2))
#         s0 = t if t % 2 != 0 else t + 1
#         if s0 < 3:
#             s0 = 3
#         s1 = int(2**1 * (s0 - 1) + 1)
#         s2 = int(2**2 * (s0 - 1) + 1)
#         s3 = int(2**3 * (s0 - 1) + 1)
#         self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=s0, padding=int(s0/2), bias=False)
#         self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=s1, padding=int(s1/2), bias=False)
#         self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=s2, padding=int(s2/2), bias=False)
#         self.conv1_4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=s3, padding=int(s3/2), bias=False)
#         self.conv1_out = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)

#         hidden_channels = max(c1 // reduction, 16)
#         self.mask = nn.Sequential(
#             nn.Conv2d(c1, hidden_channels, kernel_size=1, bias=False),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden_channels, c1, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         f2 = self.gap(x)
#         transform = f2.view(f2.size(0), 1, -1)
#         c0 = self.conv1_1(transform) # shape: (B, 1, c2)
#         c1 = self.conv1_2(transform) 
#         c2 = self.conv1_3(transform)
#         c3 = self.conv1_4(transform)
#         c_out = torch.cat([c0, c1, c2, c3], dim=1)
#         fuse = self.conv1_out(c_out)
#         fuse = fuse.transpose(1, 2).unsqueeze(-1)
#         attn = self.mask(fuse)

#         return x * attn

# class CPKS(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)  
#         self.cv1 = nn.Sequential(
#             nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(2 * self.c),
#             nn.SiLU(inplace=True)
#         )
#         self.m = nn.ModuleList(AODC(self.c, shortcut=shortcut) for _ in range(n))

#         self.cv2 = nn.Sequential(
#             nn.Conv2d((2 + n) * self.c, c2, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True)
#         )

#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m) 
#         return self.cv2(torch.cat(y, 1))

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from ultralytics.nn.modules.conv import Conv
# from .block import C2f

# class GhostModule(nn.Module):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels * (ratio - 1)

#         self.primary_conv = nn.Sequential(
#             nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
#             nn.BatchNorm2d(init_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )

#         self.cheap_operation = nn.Sequential(
#             nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
#             nn.BatchNorm2d(new_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )

#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         out = torch.cat([x1, x2], dim=1)
#         return out[:, :self.oup, :, :]


# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v

# def hard_sigmoid(x, inplace: bool = False):
#     if inplace:
#         return x.add_(3.).clamp_(0., 6.).div_(6.)
#     else:
#         return F.relu6(x + 3.) / 6.

# class SqueezeExcite(nn.Module):
#     def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
#                  act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
#         super(SqueezeExcite, self).__init__()
#         self.gate_fn = gate_fn
#         reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
#         self.act1 = act_layer(inplace=True)
#         self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

#     def forward(self, x):
#         x_se = self.avg_pool(x)
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act1(x_se)
#         x_se = self.conv_expand(x_se)
#         x = x * self.gate_fn(x_se)
#         return x    

# class GhostBottleneck(nn.Module):
#     """ Ghost bottleneck w/ optional SE"""

#     def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
#                  stride=1, act_layer=nn.ReLU, se_ratio=0., use_shortcut=True):
#         super(GhostBottleneck, self).__init__()
#         has_se = se_ratio is not None and se_ratio > 0.
#         self.stride = stride

#         # Point-wise expansion
#         self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
#         self.use_shortcut = use_shortcut and (in_chs == out_chs and self.stride == 1)

#         # Depth-wise convolution
#         if self.stride > 1:
#             self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
#                              padding=(dw_kernel_size-1)//2,
#                              groups=mid_chs, bias=False)
#             self.bn_dw = nn.BatchNorm2d(mid_chs)

#         # Squeeze-and-excitation
#         if has_se:
#             self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
#         else:
#             self.se = None

#         # Point-wise linear projection
#         self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
#         # shortcut
#         # if (in_chs == out_chs and self.stride == 1):
#         #     self.shortcut = nn.Sequential()
#         if self.use_shortcut:
#             self.shortcut = nn.Sequential()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
#                        padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
#                 nn.BatchNorm2d(in_chs),
#                 nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_chs),
#             )


#     def forward(self, x):
#         residual = x

#         # 1st ghost bottleneck
#         x = self.ghost1(x)

#         # Depth-wise convolution
#         if self.stride > 1:
#             x = self.conv_dw(x)
#             x = self.bn_dw(x)

#         # Squeeze-and-excitation
#         if self.se is not None:
#             x = self.se(x)

#         # 2nd ghost bottleneck
#         x = self.ghost2(x)
        
#         x += self.shortcut(residual)
#         return x
    
# class C2fGhost(C2f):

#     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)

#         self.m = nn.ModuleList(
#             # GhostBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
#             GhostBottleneck(
#                 in_chs=self.c,            
#                 mid_chs=self.c,           
#                 out_chs=self.c,           
#                 dw_kernel_size=3,         
#                 stride=1,                 
#                 act_layer=nn.SiLU,        
#                 se_ratio=0.,              
#                 use_shortcut=shortcut     
#             ) 
#             for _ in range(n)
#         )
    

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import PSABlock, C3k
from .block import C2f

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., use_shortcut=True):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        self.use_shortcut = use_shortcut and (in_chs == out_chs and self.stride == 1)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        # if (in_chs == out_chs and self.stride == 1):
        #     self.shortcut = nn.Sequential()
        if self.use_shortcut:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x
    
class C2fGhost(C2f):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            # GhostBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            GhostBottleneck(
                in_chs=self.c,            
                mid_chs=self.c,           
                out_chs=self.c,           
                dw_kernel_size=3,         
                stride=1,                 
                act_layer=nn.SiLU,        
                se_ratio=0.,              
                use_shortcut=shortcut     
            ) 
            for _ in range(n)
        )

class C3k2_Ghost(C2f):
    """C3k2 module sử dụng GhostBottleneck làm khối lõi."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        se_ratio: float = 0.0,  # Tỉ lệ SE Attention cho GhostBottleneck (mặc định 0 - tắt SE)
    ):
        """Initialize C3k2_Ghost module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
            se_ratio (float): Ratio for Squeeze-and-Excitation inside GhostBottleneck.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        
        # Khởi tạo khối GhostBottleneck thay thế cho Bottleneck chuẩn
        # Kênh vào: self.c, Kênh trung gian (mid): self.c, Kênh ra: self.c
        self.m = nn.ModuleList(
            nn.Sequential(
                GhostBottleneck(
                    in_chs=self.c, 
                    mid_chs=self.c, 
                    out_chs=self.c, 
                    se_ratio=se_ratio, 
                    use_shortcut=shortcut
                ),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else GhostBottleneck(
                in_chs=self.c, 
                mid_chs=self.c, 
                out_chs=self.c, 
                se_ratio=se_ratio, 
                use_shortcut=shortcut
            )
            for _ in range(n)
        )