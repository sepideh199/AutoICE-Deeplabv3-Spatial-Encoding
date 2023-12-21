from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import ASPP

###############################################
###############################################
class ResNetASPP(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if kwargs['pretrained'] == True:
            weights=ResNet18_Weights.IMAGENET1K_V1
        else:
            weights=None

        self.encoder = resnet18(weights=weights)

        self.encoder.conv1 = nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        return_layers = {"layer3": "out"}
        self.encoder = IntermediateLayerGetter(self.encoder, return_layers=return_layers)
    
        # create the head:
        self.sic_decoder = nn.Sequential(ASPP(in_channels=256, atrous_rates = [12, 24, 36], out_channels = 32),
                                        nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, kwargs['n_classes']['SIC'], 1)
                                        )

        self.sod_decoder = nn.Sequential(ASPP(in_channels=256, atrous_rates = [12, 24, 36], out_channels = 32),
                                        nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, kwargs['n_classes']['SOD'], 1))

        self.floe_decoder = nn.Sequential(ASPP(in_channels=256, atrous_rates = [12, 24, 36], out_channels = 32),
                                        nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, kwargs['n_classes']['FLOE'], 1))

        # ASPP has one global average pooling that messes things up 
        # in case we want to change the input size (full raster prediction)
        avgpool_replacer = nn.AvgPool2d(2,2)
        for dec in [self.sic_decoder, self.sod_decoder, self.floe_decoder]:
            if isinstance(dec[0].convs[-1][0], nn.AdaptiveAvgPool2d):
                dec[0].convs[-1][0] = avgpool_replacer
            else:
                print('Check the model! Is there an AdaptiveAvgPool2d somewhere?')
        
    def forward(self, x:Tensor) -> Tensor:
        
        input_shape = x.shape[-2:]

        features = self.encoder(x)['out']

        y_hat = self.sic_decoder(features)         
        sic = F.interpolate(y_hat, size=input_shape, mode="bilinear", align_corners=False)

        y_hat = self.sod_decoder(features)         
        sod = F.interpolate(y_hat, size=input_shape, mode="bilinear", align_corners=False)

        y_hat = self.floe_decoder(features)         
        floe = F.interpolate(y_hat, size=input_shape, mode="bilinear", align_corners=False)

        return {'SIC': sic,
                'SOD': sod,
                'FLOE': floe}