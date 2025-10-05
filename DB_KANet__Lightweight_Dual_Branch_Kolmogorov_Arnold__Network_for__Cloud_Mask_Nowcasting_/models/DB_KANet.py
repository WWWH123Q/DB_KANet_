
import numbers
import os
import sys
from utils import *
from configs.config_setting import setting_config

config = setting_config
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight
        return torch.nn.functional.sigmoid(out)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


from torch.nn import init

import math
import torch.nn.functional as F
from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.nn.functional import conv3d, conv2d, conv1d




class ChebyshevFunction(nn.Module):
    def __init__(self, degree: int = 4):
        super(ChebyshevFunction, self).__init__()
        self.degree = degree

    def forward(self, x):
        chebyshev_polynomials = [torch.ones_like(x), x]
        for n in range(2, self.degree):
            chebyshev_polynomials.append(2 * x * chebyshev_polynomials[-1] - chebyshev_polynomials[-2])
        return torch.stack(chebyshev_polynomials, dim=-1)


class SplineConv2D(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 init_scale: float = 0.1,
                 padding_mode: str = "zeros",
                 **kw
                 ) -> None:
        self.init_scale = init_scale
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode,
                         **kw
                         )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FastKANConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 grid_min: float = -2.,
                 grid_max: float = 2.,
                 num_grids: int = 4,
                 use_base_update: bool = True,
                 base_activation=F.relu,  # silu
                 spline_weight_init_scale: float = 0.1,
                 padding_mode: str = "zeros",
                 kan_type: str = "RBF",
                 ) -> None:

        super().__init__()
        if kan_type == "Chebyshev":
            self.rbf = ChebyshevFunction(num_grids)

        self.spline_conv = SplineConv2D(in_channels * num_grids,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        groups,
                                        bias,
                                        spline_weight_init_scale,
                                        padding_mode)

        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dilation,
                                       groups,
                                       bias,
                                       padding_mode)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_rbf = self.rbf(x.view(batch_size, channels, -1)).view(batch_size, channels, height, width, -1)
        x_rbf = x_rbf.permute(0, 4, 1, 2, 3).contiguous().view(batch_size, -1, height, width)

        # Apply spline convolution
        ret = self.spline_conv(x_rbf)

        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret = ret + base

        return ret

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )




    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class Luo3DCNN_KAN_Net_1(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)
# ------------------------------------------------------------------------------------------------------------------

    # def __init__(self,input_channels,input_frames):
    def __init__(self,input_frames,output_frames):
        super(Luo3DCNN_KAN_Net_1, self).__init__()
        self.input_frames = input_frames
        self.out_frames=output_frames
        self.conv1_1 = nn.Conv2d(self.input_frames, self.input_frames, (3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2_1 = nn.Sequential(
            FastKANConvLayer(in_channels=self.input_frames,
                             out_channels=self.out_frames,
                             kernel_size=(3, 3),
                             stride=2,
                             padding=1,
                             kan_type="Chebyshev"
                             ),
                            nn.BatchNorm2d(self.out_frames),
                            nn.ReLU(),
                    )
        self.conv2_2 = nn.Conv2d(self.input_frames,self.out_frames,kernel_size=(3, 3),
                             stride=2,
                             padding=1,)
# ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.gelu(self.conv2_1(x))
        return x



class Luo3DCNN_KAN_Net_2(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)
# ------------------------------------------------------------------------------------------------------------------

    def __init__(self,input_frames,output_frames):
        super(Luo3DCNN_KAN_Net_2, self).__init__()

        self.input_frames = input_frames
        self.out_frames=output_frames


        self.conv1_2 = nn.Conv2d(self.input_frames, self.input_frames, (3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 长宽加倍
            FastKANConvLayer(in_channels=self.input_frames,
                             out_channels=self.out_frames,
                             kernel_size=(3, 3),
                             stride=1,
                             padding=1,
                             kan_type="Chebyshev"
                             ),
            nn.BatchNorm2d(self.out_frames),
            nn.ReLU(),
        )
        self.conv2_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 长宽加倍
            nn.Conv2d(in_channels=self.input_frames,
                             out_channels=self.out_frames,
                             kernel_size=(3, 3),
                             stride=1,
                             padding=1,
                             ),
            nn.BatchNorm2d(self.out_frames),
            nn.ReLU(),
        )

# ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):

        batch_size, frames, height, width = x.shape
        x = F.relu(self.conv1_2(x))
        x = F.gelu(self.conv2_2(x))
        return x


class en_PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_frames = input_dim
        self.output_frames = output_dim
        self.en=Luo3DCNN_KAN_Net_1(self.input_frames,self.output_frames)

    def forward(self, x):
        return self.en(x)


class de_PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_frames = input_dim
        self.output_frames = output_dim
        self.de = Luo3DCNN_KAN_Net_2(self.input_frames, self.output_frames)

    def forward(self, x):
        return self.de(x)


class InnovativeCloBlock(nn.Module):
    def __init__(self, global_dim, local_dim, kernel_size, pool_size, head, qk_scale=None, drop_path_rate=0.0):
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.head = head

        self.norm = nn.LayerNorm(global_dim + local_dim)

        # global branch
        self.global_head = int(self.head * self.global_dim / (self.global_dim + self.local_dim))
        self.fc1 = nn.Linear(global_dim, global_dim * 3)
        self.pool1 = nn.AvgPool2d(pool_size)
        self.pool2 = nn.AvgPool2d(pool_size)
        self.qk_scale = qk_scale or global_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        # local branch
        self.local_head = int(self.head * self.local_dim / (self.global_dim + self.local_dim))
        self.fc2 = nn.Linear(local_dim, local_dim * 3)
        self.qconv = nn.Conv2d(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
                               padding=kernel_size // 2, groups=local_dim // self.local_head)
        self.kconv = nn.Conv2d(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
                               padding=kernel_size // 2, groups=local_dim // self.local_head)
        self.vconv = nn.Conv2d(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
                               padding=kernel_size // 2, groups=local_dim // self.local_head)
        self.fc3 = nn.Conv2d(local_dim // self.local_head, local_dim // self.local_head, 1)
        self.swish = nn.SiLU()  # Swish activation
        self.fc4 = nn.Conv2d(local_dim // self.local_head, local_dim // self.local_head, 1)
        self.tanh = nn.Tanh()

        # time branch
        self.time_fc = nn.Linear(global_dim + local_dim, global_dim + local_dim)
        self.time_norm = nn.LayerNorm(global_dim + local_dim)

        # fuse
        self.fc5 = nn.Conv2d(global_dim + local_dim, global_dim + local_dim, 1)

    def forward(self, x):
        identity = x

        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        # print('x.shape: ', x.shape)
        # x = self.norm(x)
        x = self.norm(x)
        x_local, x_global = torch.split(x, [self.local_dim, self.global_dim], dim=-1)

        # global branch
        global_qkv = self.fc1(x_global)
        global_qkv = rearrange(global_qkv, 'b n (m h c) -> m b h n c', m=3, h=self.global_head)
        global_q, global_k, global_v = global_qkv[0], global_qkv[1], global_qkv[2]
        global_k = rearrange(global_k, 'b h (n1 n2) c -> b (h c) n1 n2', n1=H, n2=W)
        global_k = self.pool1(global_k)
        global_k = rearrange(global_k, 'b (h c) n1 n2 -> b h (n1 n2) c', h=self.global_head)
        global_v = rearrange(global_v, 'b h (n1 n2) c -> b (h c) n1 n2', n1=H, n2=W)
        global_v = self.pool1(global_v)
        global_v = rearrange(global_v, 'b (h c) n1 n2 -> b h (n1 n2) c', h=self.global_head)
        attn = (global_q @ global_k.transpose(-2, -1)) * self.qk_scale
        attn = self.softmax(attn)
        x_global = attn @ global_v
        x_global = rearrange(x_global, 'b h (n1 n2) c -> b (h c) n1 n2', n1=H, n2=W)

        # local branch
        local_qkv = self.fc2(x_local)
        local_qkv = rearrange(local_qkv, 'b (n1 n2) (m h c) -> m (b h) c n1 n2', m=3, n1=H, n2=W, h=self.local_head)
        local_q, local_k, local_v = local_qkv[0], local_qkv[1], local_qkv[2]
        local_q = self.qconv(local_q)
        local_k = self.kconv(local_k)
        local_v = self.vconv(local_v)
        attn = local_q * local_k
        attn = self.fc4(self.swish(self.fc3(attn)))
        attn = self.tanh(attn / (self.local_dim ** -0.5))
        x_local = attn * local_v
        x_local = rearrange(x_local, '(b h) c n1 n2 -> b (h c) n1 n2', b=B)

        # time branch
        x_time = torch.cat([x_local, x_global], dim=1)  # x_time shape: [B, local_dim + global_dim, H, W]
        x_time = rearrange(x_time, 'b c h w -> b (h w) c')
        x_time = self.time_fc(x_time)
        x_time = self.time_norm(x_time)
        x_time = rearrange(x_time, 'b (h w) c -> b c h w', h=H, w=W)

        # Fuse
        x = torch.cat([x_local, x_global], dim=1)  # x shape: [B, local_dim + global_dim, H, W]
        x = self.fc5(x)  # fc5 input channels: local_dim + global_dim
        out = identity + x
        return out


class Attention(nn.Module):
    def __init__(self, input_dim, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        # 使用 FastKANConvLayer 替换 qkv 的生成
        # self.qkv = FastKANConvLayer(
        #     in_channels=dim,
        #     out_channels=dim * 3,  # 生成 q, k, v
        #     kernel_size=3,  # 1x1 卷积
        #     stride=1,
        #     padding=1,
        #     kan_type='Chebyshev',  # 可选的基函数类型：RBF, Fourier, Chebyshev, BSpline
        #     grid_min=-2., grid_max=2., num_grids=8  # 根据任务调整
        # )
        # self.qkv_2 = FastKANConvLayer(
        #     in_channels=3*dim,
        #     out_channels=dim * 3,  # 生成 q, k, v
        #     kernel_size=1,  # 1x1 卷积
        #     stride=1,
        #     padding=0,
        #     kan_type='Fourier',  # 可选的基函数类型：RBF, Fourier, Chebyshev, BSpline
        #     grid_min=-2., grid_max=2., num_grids=8  # 根据任务调整
        # )

        # Depthwise convolution for qkv
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)

        # Output projection
        self.project_out = nn.Conv2d(dim, input_dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 使用 FastKANConvLayer 生成 q, k, v
        qkv = self.qkv_dwconv(self.qkv(x))
        # qkv = self.qkv_2(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape q, k, v for multi-head attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize q and k
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.nn.functional.sigmoid(attn)

        # Apply attention to values
        out = (attn @ v)

        # Reshape output back to original shape
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Project output
        out = self.project_out(out)

        return out


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

        # 计算每个方向（height/width）的编码维度
        self.dim_per_dir = (channels + 1) // 2  # 确保奇数通道也能平分

        # 生成 height 的位置编码
        position_h = torch.arange(height).float().unsqueeze(1)  # (height, 1)
        div_term_h = torch.exp(torch.arange(0, self.dim_per_dir, 2).float() *
                              (-torch.log(torch.tensor(10000.0)) / self.dim_per_dir))
        pos_enc_h = torch.zeros(height, self.dim_per_dir)  # (height, dim_per_dir)
        pos_enc_h[:, 0::2] = torch.sin(position_h * div_term_h)
        pos_enc_h[:, 1::2] = torch.cos(position_h * div_term_h)

        # 生成 width 的位置编码
        position_w = torch.arange(width).float().unsqueeze(1)  # (width, 1)
        div_term_w = torch.exp(torch.arange(0, self.dim_per_dir, 2).float() *
                              (-torch.log(torch.tensor(10000.0)) / self.dim_per_dir))
        pos_enc_w = torch.zeros(width, self.dim_per_dir)  # (width, dim_per_dir)
        pos_enc_w[:, 0::2] = torch.sin(position_w * div_term_w)
        pos_enc_w[:, 1::2] = torch.cos(position_w * div_term_w)

        # 合并 height 和 width 的位置编码，并截断到通道数
        self.pos_enc_h = nn.Parameter(pos_enc_h[:, :self.dim_per_dir], requires_grad=False)  # (height, dim_per_dir)
        self.pos_enc_w = nn.Parameter(pos_enc_w[:, :self.dim_per_dir], requires_grad=False)  # (width, dim_per_dir)

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # 扩展位置编码以匹配空间维度
        pos_enc_h = self.pos_enc_h.unsqueeze(1).expand(-1, width, -1)  # (height, width, dim_per_dir)
        pos_enc_w = self.pos_enc_w.unsqueeze(0).expand(height, -1, -1)  # (height, width, dim_per_dir)

        # 合并 height 和 width 的位置编码，并截断到通道数
        pos_enc = torch.cat([pos_enc_h, pos_enc_w], dim=-1)  # (height, width, 2*dim_per_dir)
        pos_enc = pos_enc[:, :, :self.channels]  # 确保通道数与输入一致
        pos_enc = pos_enc.permute(2, 0, 1).unsqueeze(0)  # (1, channels, height, width)

        # 将位置编码加到输入上
        return x + pos_enc

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, dim, num_heads,height, width):
        super(TransformerBlock, self).__init__()

        # 2D 位置编码
        self.pos_encoding = PositionalEncoding2D(dim, height, width)

        self.norm1 = LayerNorm(input_dim)
        self.conv = nn.Conv2d(input_dim, dim, kernel_size=3, stride=1, padding=1)

        # Replace Attention with FastKANConvLayer
        # self.kan_attn = FastKANConvLayer(
        #     in_channels=dim,
        #     out_channels=dim,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     kan_type="Chebyshev"  # You can choose other types like "RBF", "Fourier", etc.
        # )
        self.attn = Attention(input_dim, dim, num_heads)

        self.norm2 = LayerNorm(input_dim)

        # Replace FeedForward with FastKANConvLayer
        self.kan_ffn = FastKANConvLayer(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            kan_type="Poly"  # You can choose other types like "RBF", "Fourier", etc. Poly81  BSpline
        )

    def forward(self, x):
        # Apply KAN-based attention
        # x = x + self.kan_attn(self.conv(self.norm1(x)))
        # x = x + self.attn(self.conv(self.norm1(x)))
        x = self.pos_encoding(x)
        x = x + self.attn(self.conv(self.norm1(x)))
        # Apply KAN-based feed-forward network
        x = x + self.kan_ffn(self.norm2(x))

        # Final convolution
        x = self.conv(x)

        return x

class  DB_KANet(nn.Module):
    
    def __init__(self, num_classes=3, input_channels=5, c_list=[2,4,8,16,32],c_list1=[10,20,40,80,160],
                split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge
        self.encoder1 = nn.Sequential(
            en_PVMLayer(input_dim=input_channels, output_dim=c_list[0])
        )
        self.encoder2 = nn.Sequential(
            InnovativeCloBlock(global_dim=16, local_dim=16, kernel_size=3, head=4, pool_size=2, drop_path_rate=0.0),
            en_PVMLayer(input_dim=c_list[0], output_dim=c_list[1])
        )
        self.encoder3 = nn.Sequential(
            InnovativeCloBlock(global_dim=24, local_dim=24, kernel_size=3, head=8, pool_size=2, drop_path_rate=0.0),
            en_PVMLayer(input_dim=c_list[1], output_dim=c_list[2])
        )
        self.encoder4 = nn.Sequential(
            InnovativeCloBlock(global_dim=32, local_dim=32, kernel_size=3, head=8, pool_size=2, drop_path_rate=0.0),
            en_PVMLayer(input_dim=c_list[2], output_dim=c_list1[3])
        )
        self.decoder1 = nn.Sequential(
            InnovativeCloBlock(global_dim=64, local_dim=64, kernel_size=3, head=8, pool_size=2, drop_path_rate=0.0),
            de_PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder2 = nn.Sequential(
            InnovativeCloBlock(global_dim=32, local_dim=32, kernel_size=3, head=8, pool_size=2, drop_path_rate=0.0),
            de_PVMLayer(input_dim=c_list[2], output_dim=c_list[1])
        )
        self.decoder3 = nn.Sequential(
            InnovativeCloBlock(global_dim=24, local_dim=24, kernel_size=3, head=8, pool_size=2, drop_path_rate=0.0),
            de_PVMLayer(input_dim=c_list[1], output_dim=c_list[0])
        )
        self.decoder4 = nn.Sequential(
            InnovativeCloBlock(global_dim=16, local_dim=16, kernel_size=3, head=4, pool_size=2, drop_path_rate=0.0),
            de_PVMLayer(input_dim=c_list[0], output_dim=c_list[0])
        )

        self.S=nn.Conv2d(c_list[0],num_classes,3,1,1)
        self.reinforcement = nn.Sequential(
            *[TransformerBlock(input_dim=num_classes, dim=num_classes, num_heads=1,height=256, width=256) for _ in range(3)],
            nn.BatchNorm2d(num_classes),
        )
        self.beta= nn.Parameter(torch.tensor(1.0,dtype=torch.float))




    def forward(self, x):
        out = self.encoder1(x)
        t1 = out
        out = self.encoder2(out)
        t2 = out
        out = self.encoder3(out)
        t3 = out
        out = self.encoder4(out)
        t4 = out

        out4 = self.decoder1(out)
        out3 = self.decoder2(out4)
        out3 = torch.add(out3, t2)
        out2 = self.decoder3(out3)
        out2 = torch.add(out2, t1)
        out1 = self.decoder4(out2)
        out1 = torch.add(out1, x[:,-1,...].unsqueeze(1))
        out0=self.S(out1)
        out00=self.reinforcement(out0)
        out00=out00*torch.sigmoid(self.beta*out00)
        return out00

