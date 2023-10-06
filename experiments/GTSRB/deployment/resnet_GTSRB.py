import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import torch


# lightning module using the resnet18 pretrained on ImageNet
class ResFour(pl.LightningModule):
    def __init__(self, lr = 0.001, weight_decay= 0.0001 , num_class = 4, *args, **kwargs):
        super().__init__()
        # save the hyperparameters
        self.save_hyperparameters()
        # load the pretrained resnet18
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        # replace the last layer by a linear layer with the right number of classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.weight.shape[1], self.hparams.num_class)
        # define the loss function
        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.test_loss = nn.CrossEntropyLoss()

        # define the metrics. One accuracy object separately for train, val and test because of epoch agregations
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_class)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_class)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_class)

    def forward(self, x):
        # called with self(x)
        return self.resnet18(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.train_loss(y_hat, y)
        # Logging the train loss metric at each step
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # Pass object accuracy rather than the value
        # log the accuracy on the training set
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.val_loss(y_hat, y)
        # Logging the val loss metric at each step
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # log the accuracy on the validation set
        self.log('val_acc', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Best practice : train loss and accuracy on step, val loss and accuracy on epoch

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.test_loss(y_hat, y)
        # Logging the test loss metric at each step
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # log the accuracy on the test set
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
