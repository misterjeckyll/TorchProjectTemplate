import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from GTSRB_dataset import GTSRBDataModule
from resnet_GTSRB import ResFour


def cli_main():
    cli = LightningCLI(
        ResFour,
        GTSRBDataModule,
        subclass_mode_data=True,
        subclass_mode_model=True,
        seed_everything_default=42,
        save_config_callback=None,

    )


if __name__ == "__main__":
    cli_main()