
import pytorch_lightning as pl
from torchmetrics import Accuracy
from utils import plot_images
import matplotlib.pyplot as plt
import wandb
from utils import plot_images


# pytorch lightning callback at the end of each epoch to report the batch prediction image to wandb
class batchReportCallback(pl.Callback):
    """
    Callback to assemble and log a visualisation
    of the model predictions at the end of each epoch from a sample of the validation set
    """
    def __init__(self, num_samples=8, grid_width=4, grid_height=2, fig_width=5, fig_height=2, fig_dir="./results"):
        super().__init__()
        self.num_samples = num_samples
        self.figure_size = (grid_width, grid_height)
        self.grid_shape = (grid_height, grid_width)
        self.fig_dir = fig_dir

    def plot_batch(self, x,y,y_hat, title="Batch Visualization"):
        # plot a grid of images from the first batch using matplotlib and subfigure
        fig = plt.figure(figsize=self.figure_size, layout='constrained')
        # add a main title to the figure
        fig.suptitle(title, fontsize=16)

        # select a sample of the batch to plot
        for index, (img, label) in enumerate(zip(x[:self.num_samples], y[:self.num_samples])):
            # convert tensor image to a numpy array
            rgb_img = img.numpy().transpose((1, 2, 0))
            # get the labels
            label_value = label.numpy()
            label_predicted = y_hat[index].argmax().numpy()
            # change color based on the prediction
            col = "green"
            if label != label_predicted:
                col = "red"

            # plot the image with its label, using the cifar list to get the label name
            plot_images(fig, index, self.grid_shape[0], self.grid_shape[1], rgb_img, label_value, col)

        return fig

    def on_validation_epoch_start(self, trainer, pl_module):
        # get the next batch of the validation set
        x, y = next(iter(trainer.datamodule.val_dataloader()))
        # get the predictions of the model
        y_hat = pl_module(x)
        # compute the accuracy
        acc = pl_module.val_accuracy(y_hat, y)

        fig = self.plot_batch(x,y,y_hat, title=f'Epoch: {trainer.current_epoch} - Accuracy: {acc}')

        # take only the decimal part of the accuracy for a cleaner filename
        acc_log = acc - acc.int()
        fig.savefig(f'{self.fig_dir}/{acc_log}_{trainer.current_epoch}.png')
        # log image to wandb
        wandb.log({"val_images": wandb.Image(fig, caption=f"batch_visu_acc:{acc}")})


