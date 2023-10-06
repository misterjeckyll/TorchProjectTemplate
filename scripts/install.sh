

conda create -y -vvv -c conda-forge -n pytorch tensorboard pillow jupyterlab
source activate pytorch
ipython kernel install --user --name=pytorch
conda export > environment.yml