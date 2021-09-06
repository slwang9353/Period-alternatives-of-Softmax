from model import Demo
import torch
from torch.nn import init
import torch.nn as nn

def initNetParams(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)


def demo_custom(args, pre_train=False, state_dir=None):
    model = Demo(**args)
    if pre_train:
        print('Loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Model loaded.')
    else:
        initNetParams(model)
        print('Model initialized.')
    return model