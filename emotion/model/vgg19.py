import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast




class vgg19(nn.Module):
    def __init__(self, in_chanel, class_out, batch_norm : bool = False):

        super(vgg19, self).__init__()
        self.ctg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        self.features = self.make_layer(in_chanel, self.ctg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, class_out),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    def make_layer(self, in_chanels : int , ctg : list, batch_norm : bool = False):
        layers: List[nn.Module] = []
        ic = in_chanels
        for layer in ctg[0]:
            if (layer == 'M'):
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            else:
                layer = cast(int, layer)

                conv2d = nn.Conv2d(ic , layer, kernel_size = (3,3), padding = 1)
                ic = layer
                
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)
