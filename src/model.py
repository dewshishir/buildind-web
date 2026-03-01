import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

class SegModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)