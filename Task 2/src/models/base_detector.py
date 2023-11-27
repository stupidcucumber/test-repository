from torch import nn
from .backbone import VGGEncoder, VGGDecoder


class MagicPoint(nn.Module):
    def __init__(self, image_height: int=224, image_width: int=224):
        super().__init__()

        self.encoder = VGGEncoder()
        self.decoder = VGGDecoder(image_height=image_height, image_width=image_width)


    def forward(self, input):
        x = self.encoder(input)
        logits = self.decoder(x)

        return logits
