# Template global project name
> Code, documentation and project structure template for a pytorch deep learning project


<img width=900 src="img/brain.png" style="margin-right: 20px">

---
📑 **Table of content**
- [👀 Overview](#-overview)
- [💻 Installation](#-installation)
- [🚀 Quick start](#-quick-start)
- [📋 Project structure](#-project-structure)
- [🔬 Technical overview](#-technical-overview)
- [📝 License](#-license)
- [💬 Contact](#-contact)
---

## 👀 Overview

This is a template pytorch lightning strucure, that can be used as a base for any Machine learning or 
Deep Learning projects. A sample experiment is available using pytorch lightning to train a road signs 
signalisation classifier, and deploy the resulting model in a prototype webapp.  


## 💻 Installation
Installation based on anaconda virtual environments.
Tested on Windows 10 Pro edition

1.Open a linux enabled command line shell like GitBash or Linux WSL, located on the project root directory

### Automated install 

Grant execution permission for the install script then run it. It will take a few minutes. 
```bash
chmod u+x ./scripts/install.sh
./scripts/install.sh
```

### Manual install

Creating a new anaconda environment from the environment file
```bash
conda env create --name pytorch --file=pytorch_environment.yml
```

For installation in another system environment (linux or mac os for example), <br>
or if there is any dependencies errors using the env file, create the necessary python environment by hand.

```bash
conda create -y -vvv -c conda-forge -n pytorch tensorboard pillow jupyterlab wandb
```
The dependency tree may take up to 30min to solve, check that you have a solid internet connection and domain access <br>
alllowed to the packages  repositories (https://anaconda.org/anaconda/repo and https://pypi.org/).

Install the virtual environment kernel for jupyter lab.
```bash
source activate pytorch
ipython kernel install --user --name=pytorch
conda export > user_environment.yml
```
## 🚀 Quick start
Open a command line shell in the project root directory
### Launch a jupyter lab notebook experiment for prototyping 
```bash
source activate pytorch
jupyter lab
```

### start a training  when model development is finished

Open a command line shell in the project root directory.

Activate the virtual environnment
```bash
source activate pytorch
```
Start the training script using a configuration file
```bash
python main.py fit -c train_config.yaml
```

To add a new parameters to the configuration file, add it with
the command line then save the generated config .yaml
```bash
python main.py fit -c train_config.yaml --train.logger=[WandbLogger, TensorBoardLogger] > config.yaml
```

```bash
python main.py fit --config train_config.yaml --ckpt_path ../checkpoints/resnetfour_epoch=001-val_loss=0.01.ckpt
```

## 📋 Project structure

The project is organized in the following way :
```
TorchProjectTemplate
├── documentation
├── Experiments
│   └── GTSRB                   # An experiment is a sub project assotiated with a single dataset
│       ├── checkpoints         # Saved training states to go back to
│       ├── data                # Separate data folder
│           └── dataset         # Cleaned ppm dataset object location 
│           └── store           # Raw data archives and files
│       ├── deployment          # Proof of concept webapp deployment code
│       ├── logs                # Training metrics logs
│       └── results             # Training result figures 
│       └── GTSRB.md            # Dataset and task description, hypothesises and conclusions 
|       └── train_config.yaml   # hyperparameters and training config file
├── img                         # Assets folder for the readme markdown 
├── scripts                     # Place here your installation and deployement shell scripts 
└── pytorch_environment.yml     # Anaconda virtual environment file
└── README.md
```

## 🔬 Technical overview

## 📝 License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">Pytorch lightning project template</span> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/misterjeckyll">William Pantry</a> is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>

As such you are allowed to :
- Share
- Adapt

For personal, research or educational purposes.

**Under the following conditions :**
- <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"> Author attribution
- <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"> No commercial use
- <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"> Use the same license

See details in the LICENSE.txt file

## 💬 Contact

For any questions or to discuss opening rights please contact us ( 🇫🇷 | 🇬🇧 ) :

<table>
<tr>
    <td>Author : </td>
    <td> </td>
</tr>
</table>