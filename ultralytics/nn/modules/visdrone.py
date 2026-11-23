import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k, C3k2

class DDM(nn.Module):
    """Chunk -> [GConv3x3 s2 -> Conv1x1 -> (MaxPool1x3 + MaxPool3x1)] || [Conv3x3 s2] -> concat -> Conv1x1."""

    def __init__(self, c1, c2, g=4):
        super().__init__()
        self.ca = c1 // 2
        self.cb = c1 - self.ca
        c_ = c2 // 2
        ga = math.gcd(self.ca, g)  # đảm bảo chia hết
        # nhánh 1
        self.gconv = Conv(self.ca, self.ca, 3, 2, g=ga)
        self.pw = Conv(self.ca, c_, 1, 1)
        self.mp_h = nn.MaxPool2d((1, 3), stride=1, padding=(0, 1))
        self.mp_v = nn.MaxPool2d((3, 1), stride=1, padding=(1, 0))
        # nhánh 2
        self.conv = Conv(self.cb, c_, 3, 2)
        # fuse
        self.fuse = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        a, b = x.split([self.ca, self.cb], dim=1)
        a = self.pw(self.gconv(a))
        a = self.mp_h(a) + self.mp_v(a)
        b = self.conv(b)
        return self.fuse(torch.cat((a, b), 1))

class FDM(nn.Module):
    def __init__(self, c1, c2, p=2, r=4, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.dwt = HaarDWT()
        self.avg = nn.AvgPool2d(p, 1, p // 2, count_include_pad=False)
        self.max = nn.MaxPool2d(p, 1, p // 2)

        cm = max(c1 // r, 4)
        self.spatial = nn.Sequential(Conv(6, cm, 1), Conv(cm, cm, 3), nn.Conv2d(cm, 3, 1))
        hidden = max((c1 + 4) // 4, 8)

        self.mlp = nn.Sequential(nn.Linear(c1 + 4, hidden), nn.SiLU(), nn.Linear(hidden, 3))

        self.alpha = nn.Parameter(torch.zeros(1))

        self.proj = Conv(4 * c1, c2, 1, 1)

        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _rms_map(self, x):
        return torch.sqrt(
            x.float().pow(2).mean(dim=1, keepdim=True)
            + self.eps
        ).to(dtype=x.dtype)

    def forward(self, x):

        ll, lh, hl, hh = self.dwt(x)
        rms = self._rms_map(ll)
        energy_lh = self._rms_map(lh)
        energy_hh = self._rms_map(hh)
        energy_hl = self._rms_map(hl)

        spatial_input = torch.cat([self.avg(rms), self.max(rms), rms, energy_lh, energy_hh, energy_hl], dim=1)

        sp = self.spatial(spatial_input)
        ll_global = torch.sqrt(
            ll.float().pow(2).mean(dim=(2, 3)) + self.eps
        ).to(dtype=ll.dtype)
        g = torch.cat([rms.mean((2, 3)), ll_global, energy_lh.mean((2, 3)), energy_hh.mean((2, 3)), energy_hl.mean((2, 3))], dim=1)
        g = self.mlp(g) * torch.sigmoid(self.alpha)
        logits = sp + g[:, :, None, None]
        beta = 2.0 * torch.sigmoid(logits)
        beta_lh, beta_hh, beta_hl = beta.chunk(3, dim=1)
        fused = torch.cat([ll, lh * beta_lh, hh * beta_hh, hl * beta_hl], dim=1)
        return self.proj(fused)



class FEM(nn.Module):
    def __init__(self, c1, c2=None, r=4, k=1):
        super().__init__()

        c2 = c1 if c2 is None else c2
        cr = max(c1 // r, 1)
    
        self.b1 = nn.Sequential(
            Conv(c1, cr, 1),
            Conv(cr, cr, k=(3, 1), s=1, p=(1, 0), g=cr),
            Conv(cr, cr, k=(1, 3), s=1, p=(0, 1), g=cr),
        )

        self.b2 = nn.Sequential(
            Conv(c1, cr, 1),
            Conv(cr, cr, k=3, s=1, g=cr)
        )

        self.b3 = nn.Sequential(
            Conv(c1, cr, 1),
            Conv(cr, cr, k=3, s=1, d=3, g=cr)
        )

        self.b4 = Conv(c1, cr, 1)

        self.fuse = Conv(4 * cr, c1, 1)

        self.skip = (
            nn.Identity()
            if int(4 * cr) == c1
            else Conv(4 * cr, c1, 1)
        )
        if k % 2 == 0:
            raise ValueError("Attention kernel k must be odd.")

        self.att = nn.Conv2d(2, c1, kernel_size=3, padding=1, bias=True)

        self.out = (
            nn.Identity()
            if c2 == c1
            else Conv(c1, c2, 1)
        )

        nn.init.zeros_(self.att.weight)
        nn.init.zeros_(self.att.bias)

    def forward(self, x):

        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        feature = torch.cat([y1, y2, y3, y4], dim=1)
        z = self.fuse(feature)
        avg = z.mean(dim=1, keepdim=True)
        max_ = z.amax(dim=1, keepdim=True)
        s = torch.cat([avg, max_], dim=1)
        logits = self.att(s)
        attention = torch.sigmoid(logits)

        feature = self.skip(feature)

        y = feature * attention + feature

        return self.out(y)

class HaarDWT(nn.Module):
    """2D Haar DWT with replicate padding for odd dimensions."""

    def forward(self, x):

        h, w = x.shape[-2:]

        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(
                x,
                (0, w % 2, 0, h % 2),
                mode="replicate"
            )

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        ll = (a + b + c + d) / 2
        lh = (a + b - c - d) / 2
        hl = (a - b + c - d) / 2
        hh = (a - b - c + d) / 2

        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    """Inverse Haar DWT, matching the convention above."""

    def forward(self, ll, lh, hl, hh, output_size=None):

        a = (ll + lh + hl + hh) / 2
        b = (ll + lh - hl - hh) / 2
        c = (ll - lh + hl - hh) / 2
        d = (ll - lh - hl + hh) / 2

        B, C, H, W = ll.shape

        x = torch.stack([a, b, c, d], dim=-1)
        x = x.reshape(B, C, H, W, 2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, C, H * 2, W * 2)

        if output_size is not None:
            h, w = output_size
            x = x[..., :h, :w]

        return x


class WaveletAttentionReconstruction(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()

        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        self.ll_refine = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),

            nn.Conv2d(c1, c1, 3, padding=1, bias=True)
        )

        # ----------------------------------
        # 2. High-frequency feature refinement
        # LH + HL + HH = 3C channels
        # ----------------------------------
        self.hf_refine = nn.Sequential(
            nn.Conv2d(
                3 * c1,
                3 * c1,
                3,
                padding=1,
                groups=3 * c1,
                bias=False
            ),
            nn.BatchNorm2d(3 * c1),
            nn.SiLU(inplace=True),

            nn.Conv2d(3 * c1, 3 * c1, 1, bias=True)
        )

        # ----------------------------------
        # 3. Joint spatial attention generator
        # Input: 4 descriptors
        # Output: 2 attention logits
        # ----------------------------------
        hidden = max(c1 // reduction, 8)

        self.attention = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, 2, 1, bias=True)
        )

    @staticmethod
    def _spatial_descriptor(x):
        magnitude = x.abs()

        avg = magnitude.mean(dim=1, keepdim=True)
        max_ = magnitude.amax(dim=1, keepdim=True)

        return torch.cat([avg, max_], dim=1)

    def forward(self, x, return_attention=False):

        original_size = x.shape[-2:]
        ll, lh, hl, hh = self.dwt(x)

        ll_e = ll + self.ll_refine(ll)

        hf = torch.cat([lh, hl, hh], dim=1)
        hf_e = hf + self.hf_refine(hf)

        lh_e, hl_e, hh_e = hf_e.chunk(3, dim=1)

        d_ll = self._spatial_descriptor(ll_e)
        d_hf = self._spatial_descriptor(hf_e)

        descriptors = torch.cat([d_ll, d_hf], dim=1)

        logits = self.attention(descriptors)

        logit_ll = logits[:, 0:1, :, :]
        logit_hf = logits[:, 1:2, :, :]

        gain_ll = 2.0 * torch.sigmoid(logit_ll)
        gain_hf = 2.0 * torch.sigmoid(logit_hf)

        ll_out = ll_e * gain_ll

        lh_out = lh_e * gain_hf
        hl_out = hl_e * gain_hf
        hh_out = hh_e * gain_hf


        y = self.idwt(
            ll_out,
            lh_out,
            hl_out,
            hh_out,
            output_size=original_size
        )

        if return_attention:
            return y, {
                "gain_ll": gain_ll,
                "gain_hf": gain_hf
            }

        return y


class WARConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.war = WaveletAttentionReconstruction(c1)

        if c1 != c2:
            self.proj = Conv(c1, c2, 1, 1)
        else:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(c1),
                nn.SiLU(inplace=True)
            )

    def forward(self, x):
        x = self.war(x)
        return self.proj(x)

class WAR_Bottleneck(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
    ):
        super().__init__()

        if g != 1:
            raise ValueError(
                "WAR_Bottleneck currently supports g=1 only."
            )

        if any(kernel != 3 for kernel in k):
            raise ValueError(
                "WAR_Bottleneck is designed to replace two 3x3 convolutions."
            )

        c_ = int(c2 * e)

        self.cv1 = WARConv(c1, c_)

        self.cv2 = WARConv(c_, c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):

        y = self.cv2(self.cv1(x))

        return x + y if self.add else y
    
class C3k_WAR(C3k):

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k=3,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            shortcut=shortcut,
            g=g,
            e=e,
            k=k,
        )

        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *[
                WAR_Bottleneck(
                    c1=c_,
                    c2=c_,
                    shortcut=shortcut,
                    g=g,
                    k=(3, 3),
                    e=1.0,
                )
                for _ in range(n)
            ]
        )
class C3k2_WAR(C3k2):

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        attn=False,
        g=1,
        shortcut=True,
    ):

        if attn:
            raise NotImplementedError(
                "C3k2_WAR does not implement PSABlock attention."
            )

        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            attn=False,
            g=g,
            shortcut=shortcut,
        )

        # Case 1: c3k=False
        # Direct WAR_Bottleneck

        if not c3k:

            self.m = nn.ModuleList([
                WAR_Bottleneck(
                    c1=self.c,
                    c2=self.c,
                    shortcut=shortcut,
                    g=g,
                    k=(3, 3),
                    e=1.0,
                )
                for _ in range(n)
            ])

        # Case 2: c3k=True
        # C3k containing WAR_Bottleneck

        else:

            self.m = nn.ModuleList([
                C3k_WAR(
                    c1=self.c,
                    c2=self.c,
                    n=2,
                    shortcut=shortcut,
                    g=g,
                    e=0.5,
                    k=3,
                )
                for _ in range(n)
            ])