import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch import einsum
from torch.nn.modules.container import ModuleList
from periodic_softmax import *

class MSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_tokens=90, score_function=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.dim, self.dropout = dim, dropout
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_tokens = num_tokens
        self.attend = score_function

        self.points = range(self.num_tokens)
        attention_offsets = {}
        self.idxs = []
        for p1 in self.points:
            for p2 in self.points:
                offset = p1 - p2
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)  # give an index to every kind of offset
                self.idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(self.heads, len(attention_offsets)))    # give every kind of offset a learnable bias (every head)
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(self.idxs).view(self.num_tokens, self.num_tokens)) # a unlearnable idx of attention bias

        self.to_qkv = nn.Linear(self.dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, self.dim),
            nn.Dropout(self.dropout)
        ) if project_out else nn.Identity()

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(
            dots + 
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        out = einsum('b h i j , b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class MLP(nn.Module):
    '''project=True then use ReLU between layers, bn or ln after the last layer'''
    def __init__(self, demensions, bn=False, ln=False, dropout=0., bias=False, project=True):
        super(MLP, self).__init__()
        self.demensions = demensions
        self.bn, self.ln, self.dropout, self.bias = bn, ln, dropout, bias
        self.project = project
        self.layers = []
        self.ac = nn.ReLU(inplace=True) if project else nn.Identity()
        for i in range(len(self.demensions) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.demensions[i], self.demensions[i + 1], bias=self.bias),
                    nn.Dropout(p=self.dropout),
                    self.ac
                )
            )
        if self.bn:
            self.layers.append(
                nn.BatchNorm1d(self.demensions[-1])
            )
        if self.ln:
            self.layers.append(
                nn.LayerNorm(self.demensions[-1])
            )
        self.mlp = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.mlp(x)


class SeqPool(nn.Module):
    '''From tokens (n, m, d) to a class token (n, d)'''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.project = MLP([dim, 1], dropout=0., ln=True)
        self.attend = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm1d(self.dim)
    
    def forward(self, x):
        score = self.project(x).squeeze()
        attn = self.attend(score)   # (n, m, 1)
        out = einsum('n m, n m d -> n d', attn, x)
        return self.bn(out)

class DropResidual(nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class EncoderBlock(nn.Module):
    def __init__(
        self, dim, num_tokens, heads, dim_heads, 
        exp_ratio, dropout=0., layer_dropout=0.,
        score_function=None
    ):
        super(EncoderBlock, self).__init__()
        self.dim, self.num_tokens = dim, num_tokens
        self.heads, self.dim_heads = heads, dim_heads
        self.exp_ratio = exp_ratio
        self.dropout, self.layer_dropout = dropout, layer_dropout
        self.score_function = score_function

        self.msa = MSA(
            self.dim, heads=self.heads, dim_head=self.dim_heads, 
            dropout=self.dropout, num_tokens=self.num_tokens,
            score_function=self.score_function
        )
        self.res_msa = DropResidual(self.msa, self.layer_dropout)
        self.ln_1 = nn.LayerNorm(self.dim)

        self.mlp = MLP([
            self.dim, self.exp_ratio * self.dim, self.dim
        ], dropout=self.dropout)
        self.res_mlp = DropResidual(self.mlp, self.layer_dropout)
        self.ln_2 = nn.LayerNorm(self.dim)

    def forward(self, x):
        x = self.res_msa(x)
        out_put = self.res_mlp(x)
        return out_put


class ConvBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, bias=False, groups=1, dropout=0.):
        super(ConvBn, self).__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.kernel, self.stride, self.padding, self.bias = kernel, stride, padding, bias
        self.groups = groups
        self.dropout = dropout

        self.conv = nn.Sequential(
            nn.Conv2d(
                self.in_channel, self.out_channel, 
                kernel_size=self.kernel, stride=self.stride, padding=self.padding, bias=self.bias, 
                groups=groups
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(self.out_channel)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Stem(nn.Module):
    def __init__(self, channels, dropout=0.):
        super(Stem, self).__init__()
        self.channels = channels
        self.dropout = dropout

        self.conv_list = nn.ModuleList()
        for num_channel in range(len(self.channels) - 1):
            self.conv_list.append(
                ConvBn(
                    self.channels[num_channel], self.channels[num_channel + 1], 
                    stride=2, dropout=self.dropout
                )
            )
    
    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        return x


class Embedding(nn.Module):
    def __init__(self, in_channel, dim, patch_size, dropout=0.):
        super(Embedding, self).__init__()
        self.in_channel, self.dim = in_channel, dim
        self.patch_size = patch_size
        self.dropout = dropout

        self.patch_embedding = ConvBn(
            self.in_channel, self.dim, kernel=self.patch_size, 
            stride=self.patch_size, padding=0, dropout=self.dropout
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = rearrange(x, 'b c h w -> b ( h w ) c')
        return x



