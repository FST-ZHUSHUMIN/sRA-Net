import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision
from src import const

class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.select = {
            '1': 'conv1_1',  # [batch_size, 64, 224, 224]
            '3': 'conv1_2',  # [batch_size, 64, 224, 224]
            '4': 'pooled_1',  # [batch_size, 64, 112, 112]
            '6': 'conv2_1',  # [batch_size, 128, 112, 112]
            '8': 'conv2_2',  # [batch_size, 128, 112, 112]
            '9': 'pooled_2',  # [batch_size, 128, 56, 56]
            '11': 'conv3_1',  # [batch_size, 256, 56, 56]
            '13': 'conv3_2',  # [batch_size, 256, 56, 56]
            '15': 'conv3_3',  # [batch_size, 256, 56, 56]
            '16': 'pooled_3',  # [batch_size, 256, 28, 28]
            '18': 'conv4_1',  # [batch_size, 512, 28, 28]
            '20': 'conv4_2',  # [batch_size, 512, 28, 28]
            '22': 'conv4_3',  # [batch_size, 512, 28, 28]
            '23': 'pooled_4',  # [batch_size, 512, 14, 14]
            '25': 'conv5_1',  # [batch_size, 512, 14, 14]
            '27': 'conv5_2',  # [batch_size, 512, 14, 14]
            '29': 'conv5_3',  # [batch_size, 512, 14, 14]
            '30': 'pooled_5',  # [batch_size , 512, 7, 7]
        }
        self.vgg = torchvision.models.vgg16(pretrained=True).features
        # for name, layer in self.vgg._modules.items():
        #     print(name, layer)


    def forward(self, x):
        ret = {}
        for name, layer in self.vgg._modules.items():
            # print(name, layer)
            x = layer(x)
            if name in self.select:
                ret[self.select[name]] = x
        return ret





# if __name__ == '__main__': 
#     a =  torchvision.models.resnet101(pretrained = True)
#     # print(a)
#     for name, layer in a._modules.items():
#         print('aaaaaaaaaa', name)
#     VGG16Extractor()