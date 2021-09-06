import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

'''Including :
    Cos-max, Siren-max, Sin-max, Sin-softmax, Sin2-max, Sin2-max-shifted
    Prenorm is costly, so better use the well performed single operation:
    Sin2maxShifted and SinSoftmax '''


class Prenorm(nn.Module):
    '''About 175 ms for (64, 8 ,196, 196) in CPU'''
    def __init__(self, eps=1e-05, dim=-1):
        super(Prenorm, self).__init__()
        self.eps, self.dim = eps, dim
    
    def forward(self, kq_maps):
        mean_map = repeat(
            kq_maps.mean(self.dim), 'b h i -> b h i j', j=kq_maps.shape[self.dim]
        )
        var_map = repeat(
            kq_maps.var(self.dim), 'b h i -> b h i j', j=kq_maps.shape[self.dim]
        )
        kq_maps = (kq_maps - mean_map) * ((var_map + self.eps) ** (- 0.5))
        return kq_maps


class Cosmax(nn.Module):
    '''Pre-norm work'''
    '''About 60 ms for (64, 8 ,196, 196) in CPU'''
    def __init__(self, dim=-1, prenorm=True, eps=1e-05):
        super(Cosmax, self).__init__()
        self.dim = dim
        self.prenorm, self.eps = prenorm, eps
        self.pre = nn.Identity() if not self.prenorm else Prenorm(self.eps, self.dim)
    
    def forward(self, input_tensor):
        input_tensor = self.pre(input_tensor)
        input_tensor = torch.cos(input_tensor)
        sum_term = repeat(
            input_tensor.sum(self.dim), 'b h i -> b h i j', j=input_tensor.shape[self.dim]
        )
        out_put = input_tensor / sum_term
        return out_put


class Sinmax(nn.Module):
    '''Should break down'''
    '''About 60 ms for (64, 8 ,196, 196) in CPU'''
    def __init__(self, dim=-1, prenorm=False, eps=1e-05):
        super(Sinmax, self).__init__()
        self.dim = dim
        self.prenorm, self.eps = prenorm, eps
        self.pre = nn.Identity() if not self.prenorm else Prenorm(self.eps, self.dim)
    
    def forward(self, input_tensor):
        input_tensor = self.pre(input_tensor)
        input_tensor = torch.sin(input_tensor)
        sum_term = repeat(
            input_tensor.sum(self.dim), 'b h i -> b h i j', j=input_tensor.shape[self.dim]
        )
        out_put = input_tensor / sum_term
        return out_put


class Sirenmax(nn.Module):
    '''Pre-norm work'''
    '''About 155 ms for (64, 8 ,196, 196) in CPU'''
    def __init__(self, dim=-1, prenorm=True, eps=1e-05):
        super(Sirenmax, self).__init__()
        self.dim = dim
        self.prenorm, self.eps = prenorm, eps
        self.pre = nn.Identity() if not self.prenorm else Prenorm(self.eps, self.dim)
    
    def forward(self, input_tensor):
        input_tensor = self.pre(input_tensor)
        input_tensor = (torch.sin(input_tensor) + 1) / (2 - 2 * torch.sin(input_tensor))
        sum_term = repeat(
            input_tensor.sum(self.dim), 'b h i -> b h i j', j=input_tensor.shape[self.dim]
        )
        out_put = input_tensor / sum_term
        return out_put


class Sin2max(nn.Module):
    '''Should break down'''
    '''About 80 ms for (64, 8 ,196, 196) in CPU'''
    def __init__(self, dim=-1, prenorm=False, eps=1e-05):
        super(Sin2max, self).__init__()
        self.dim = dim
        self.prenorm, self.eps = prenorm, eps
        self.pre = nn.Identity() if not self.prenorm else Prenorm(self.eps, self.dim)
    
    def forward(self, input_tensor):
        input_tensor = self.pre(input_tensor)
        input_tensor = torch.sin(input_tensor) ** 2
        sum_term = repeat(
            input_tensor.sum(self.dim), 'b h i -> b h i j', j=input_tensor.shape[self.dim]
        )
        out_put = input_tensor / sum_term
        return out_put


class Sin2maxShifted(nn.Module):
    '''Should work better alone'''
    '''About 95 ms for (64, 8 ,196, 196) in CPU'''
    def __init__(self, dim=-1, prenorm=False, eps=1e-05):
        super(Sin2maxShifted, self).__init__()
        self.dim = dim
        self.prenorm, self.eps = prenorm, eps
        self.pre = nn.Identity() if not self.prenorm else Prenorm(self.eps, self.dim)
    
    def forward(self, input_tensor):
        input_tensor = self.pre(input_tensor)
        input_tensor = torch.sin(input_tensor + 0.25 * np.pi) ** 2
        sum_term = repeat(
            input_tensor.sum(self.dim), 'b h i -> b h i j', j=input_tensor.shape[self.dim]
        )
        out_put = input_tensor / sum_term
        return out_put


class SinSoftmax(nn.Module):
    '''Should work better alone'''
    '''About 55 ms for (64, 8 ,196, 196) in CPU'''
    def __init__(self, dim=-1, prenorm=False, eps=1e-05):
        super(SinSoftmax, self).__init__()
        self.dim = dim
        self.prenorm, self.eps = prenorm, eps
        self.pre = nn.Identity() if not self.prenorm else Prenorm(self.eps, self.dim)
        self.softmax = nn.Softmax(dim=self.dim)
    
    def forward(self, input_tensor):
        input_tensor = self.pre(input_tensor)
        input_tensor = torch.sin(input_tensor)
        return self.softmax(input_tensor)


# from torch.autograd import Variable
# import torch
# x = Variable(torch.randn(64, 8, 196, 196), requires_grad=True)
# model = Prenorm(dim=-1)
# with torch.autograd.profiler.profile() as prof:
#      y = model(x)
# # NOTE: some columns were removed for brevity
# print(prof)


