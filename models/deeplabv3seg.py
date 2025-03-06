from lightning.pytorch import LightningModule
import torch.nn as nn
from torch.nn import functional as F
import torch
import torchmetrics
from torchmetrics import MetricCollection, F1Score, JaccardIndex
import torchvision.models as models

class DeepLabV3(LightningModule):
    def __init__(self, learning_rate=1e-3, num_classes=6, class_labels=None):
        super().__init__()
        self.save_hyperparameters()

        # Load the DeepLabV3 model from torchvision
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

        self.metric = MetricCollection({
            "avg_f1": F1Score('multiclass', num_classes=num_classes, average='macro', ignore_index=0),
            "avg_iou": JaccardIndex('multiclass', num_classes=num_classes, average='macro', ignore_index=0),
            "f1": torchmetrics.ClasswiseWrapper(
                F1Score('multiclass', num_classes=num_classes, average=None),
                labels=class_labels, prefix="f1_"
            ),
            "iou": torchmetrics.ClasswiseWrapper(
                JaccardIndex('multiclass', num_classes=num_classes, average=None),
                labels=class_labels, prefix="iou_"
            ),
        })

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        # y = y.squeeze(1) # Remove the channel dimension
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=False)
        preds = torch.argmax(logits, dim=1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        # y = y.squeeze(1)  # Remove the channel dimension
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self.metric(preds, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(1)  # Remove the channel dimension
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self.metric(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=False)
