from lightning.pytorch import LightningModule
import torch.nn as nn
from torch.nn import functional as F
import torch
import torchmetrics
from torchmetrics import MetricCollection, F1Score, JaccardIndex  # Added imports for MetricCollection, F1Score, and JaccardIndex

class UNetSeg(LightningModule):
    def __init__(self, learning_rate=1e-3, num_classes=6, class_labels=None):
        super().__init__()
        self.save_hyperparameters()

        # Define the U-Net model for segmentation
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.metric = MetricCollection({
            "avg_f1": F1Score('multiclass', num_classes=num_classes, average='macro', ignore_index=0),
            "avg_iou": JaccardIndex('multiclass', num_classes=num_classes, average='macro', ignore_index=0),
            "f1": torchmetrics.ClasswiseWrapper(F1Score('multiclass', num_classes=num_classes, average=None), labels=class_labels, prefix="f1_"),
            "iou": torchmetrics.ClasswiseWrapper(JaccardIndex('multiclass', num_classes=num_classes, average=None), labels=class_labels, prefix="iou_"),
        })

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        center = self.center(self.pool4(enc4))
        dec4 = self.up4(center)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.final(dec1)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        metrics = self.metric(logits, y)  # Update metrics
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        metrics = self.metric(logits, y)  # Update metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        metrics = self.metric(logits, y)  # Update metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)