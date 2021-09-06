from layers import *
from periodic_softmax import *
import torch
import torch.nn as nn



class Demo(nn.Module):
    def __init__(
        self, 
        in_h=224, stem_channels=[3, 32, 64], patch_size=4, num_blocks=2, 
        dim=512, dropout=0., layer_dropout=0., 
        heads=8, dim_heads=64, att_mlp_exp=4, score_function=None, 
        fc_dim=2048, num_class=100
    ):
        super(Demo, self).__init__()
        self.stem_channels = stem_channels
        self.dropout, self.layer_dropout = dropout, layer_dropout
        self.dim, self.patch_size = dim, patch_size
        self.num_tokens = ((in_h // (2 * (len(stem_channels) - 1))) // patch_size) ** 2
        self.heads, self.dim_heads = heads, dim_heads
        self.exp_ratio = att_mlp_exp
        self.score_function = score_function
        self.fc_dim, self.num_class = fc_dim, num_class

        self.stem = Stem(self.stem_channels, dropout=self.dropout)
        self.embedding = Embedding(self.stem_channels[-1], self.dim, self.patch_size, dropout=self.dropout)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(EncoderBlock(
                self.dim, self.num_tokens, self.heads, self.dim_heads, 
                self.exp_ratio, dropout=self.dropout, layer_dropout=self.layer_dropout,
                score_function=self.score_function
            ))
        self.seqpool = SeqPool(self.dim)
        self.fc = nn.Sequential(
            MLP([self.dim, self.fc_dim], bn=True, dropout=self.dropout),
            nn.Linear(self.fc_dim, self.num_class)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.seqpool(x)
        out_put = self.fc(x)
        return out_put

# input_tensor = torch.randn(32, 3, 224, 224)
# model = Demo( 
        # 224, stem_channels=[3, 32, 64], patch_size=4, num_blocks=2, 
        # dim=512, dropout=0.3, layer_dropout=0.3, 
        # heads=8, dim_heads=64, att_mlp_exp=4, score_function=SinSoftmax(dim=-1), 
        # fc_dim=2048, num_class=100
#     )
# out_put = model(input_tensor)