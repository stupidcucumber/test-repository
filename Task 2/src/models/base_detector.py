from torch import nn
from .model_parts import VGGBackbone, Decoder


class MagicPoint(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = VGGBackbone()
        self.decoder = Decoder()


    def forward(self, input):
        x = self.encoder(input)
        logits = self.decoder(x)

        return logits
