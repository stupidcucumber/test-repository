'''
    All rights reserved. Copyright belongs to the Ihor Kostiuk, 2023.
'''
from torch import nn


class VGGBlock(nn.Module):
    '''
        Part of the VGG Backbone.
    '''
    def __init__(self, features_in: int, features_out: int, conv_num: int):
        super().__init__()

        self.conv1 = nn.Conv2d(features_in, features_out, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.rest_conv = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(features_out, features_out, kernel_size=(3, 3), padding=1), 
                nn.ReLU(),
                nn.BatchNorm2d(features_out)
                ) for _ in range(conv_num - 1)]
        )
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.rest_conv(x)

        return self.max_pool2d(x)
        

class VGGEncoder(nn.Module):
    '''
        Encoder for the MagicPoint model.
    '''
    def __init__(self):
        super().__init__()
        self.vgg_block1 = VGGBlock(features_in=1, features_out=64, conv_num=3)
        self.vgg_block2 = VGGBlock(features_in=64, features_out=128, conv_num=3)
        self.vgg_block3 = VGGBlock(features_in=128, features_out=65, conv_num=3)


    def forward(self, input):
        x = self.vgg_block1(input)
        x = self.vgg_block2(x)
        logits = self.vgg_block3(x)

        return logits


class VGGDecoder(nn.Module):
    '''
        Decoder for the MagicPoint model.
    '''
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.height = image_height
        self.width = image_width
        self.softmax = nn.Softmax(dim=1)


    def forward(self, logits):
        logits = self.softmax(logits)
        logits = logits[:, :-1, :, :] # Dropping the last dimmention (the one that contains which represents no detection)
        logits = logits.view(-1, self.height, self.width)
        return logits