import torch
from torch import nn


class VGGBackbone(nn.Module):
    '''
            Backbone part of the MagicPoint and SuperPoint.
    '''
    def __init__(self):
        super(VGGBackbone, self).__init__()

        self.relu = nn.ReLU()
        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        
        self.conv_64_1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv_64_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv_64_4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_128_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv_128_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_128_3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.conv_128_4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)


    def forward(self, image):
        # First block
        x = self.conv_64_1(image)
        x = self.relu(x)
        x = self.bn_64(x)

        # Second block
        x = self.conv_64_2(x)
        x = self.relu(x)
        x = self.bn_64(x)

        x = self.max_pool_1(x)
        # Third block
        x = self.conv_64_3(x)
        x = self.relu(x)
        x = self.bn_64(x)

        # fourth block
        x = self.conv_64_4(x)
        x = self.relu(x)
        x = self.bn_64(x)

        x = self.max_pool_2(x)
        # Fifth block
        x = self.conv_128_1(x)
        x = self.relu(x)
        x = self.bn_128(x)

        # Sixth block
        x = self.conv_128_2(x)
        x = self.relu(x)
        x = self.bn_128(x)

        x = self.max_pool_3(x)
        # Seventh block
        x = self.conv_128_3(x)
        x = self.relu(x)
        x = self.bn_128(x)

        # Eights block
        x = self.conv_128_4(x)
        x = self.relu(x)
        x = self.bn_128(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.relu = nn.ReLU()
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_65 = nn.BatchNorm2d(65)
        
        self.conv_256_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv_65_1 = nn.Conv2d(256, 65, kernel_size=(1, 1))

        self.channelwise_softmax = nn.Softmax(dim=1)


    def forward(self, image):
        x = self.conv_256_1(image)
        x = self.relu(x)
        x = self.bn_256(x)

        x = self.conv_65_1(x)
        x = self.relu(x)
        x = self.bn_65(x)

        logits = x
        logits = logits[:, :-1, :, :]
        logits = logits.view(-1, 8 * x.shape[2], 8 * x.shape[3])

        x = self.channelwise_softmax(x)
        x = x[:, :-1, :, :]
        scores = x.view(-1, 8 * x.shape[2], 8 * x.shape[3])

        return logits, scores