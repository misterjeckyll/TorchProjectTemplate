# Formation pytorch
>Alexi lechervy - enseignant chercheur GREYC
> 
>02/06/2023
> 
> #Pytorch  #PytorchLightning #formation 

Formation doctorale sur l'ecosysteme pytorch et particulierement la librairie pytorch Lightning

```toc
```

## Ecosystème pytorch

### Pytorch : librairie bas niveau similaire numpy
Portage sur GPU, adapté pour faire des nouvelles fonctionnalité par les experts
**interet :** control fin

### Pytorch Lightning | Lightning Fabric 
Librairie de niveau intermediaire 
Adapté aux chercheurs et ingénieurs machine learning
gestion multi GPU automatique, logging
**Interet :** gagner du temps

pytorch lightning : pipeline assez strict

### Lightning Flash
très haut niveau
Plein de baseline et pipeline déja fait sur plein de taches standard
adapté débutant, très simple à prendre en mains.

**Recommendation :** commencer pytorch lightning d'abord et descendre au besoin

### Torchvision
Outils utile au traitement d'image, boite à outils pour la vision.
ex : les transformations classiques
transformation specifique pas forcement dedans. Ex qui donne plein de bonnes performance en data augmentation : prendre deux image, faire la moyenne des deux le nouveau label est la moyenne des deux. librairie timm qui fait ça sinon

Les modèles classiques (resNet), les bases standards (IMAGENet, mnist, Cifar)

### TorchMetrics
Contient la plupart des principales métriques
Pour l'utiliser, on l'ajoute à l'entrainnement ou faire à coté sur un autre thread pour ne pas ralentir l'entrainnement. Donnera dans les logs la ou les metriques choisies.

### Atouts de Torch Lightning
- Formalisme standard qui impose des bonnes façon de programmer
- Flexible via pytorch pour adapter les pipelines d'entrainement
Gros problème deep learning : difficile de reproduire les resultats à cause des générateurs aléatoire différents. Sur GPU certaines opérations sont un peu aléatoire aussi
Les pipelines d'entrainnements permettent de résoudre ces problèmes.
- Monitoring : wandB, optuna ou tensorboard facilement ajoutable
- Scalabilité TPU / MultiGPU et clusters de calculs

## Les modèles LightningModule
Definir des classes pour structurer le code
- herite de lightningModule
- Permet de definir tout ce qui est nécéssaire pour l'entrainnement
	- **\__init__** : 
	- **training_step** : pas besoin de faire la boucle à la main, ici forward et calcul fonction de cout
	- **validation_step** : quoi faire pour une etape pour la validation, forward sur le réseau et calcul de metriques
	- **test_step** : calcul d'une metrique
	- **predict_step** : la boucle de prediction
	- **configure_optimizers** : les optimiseurs et leurs parametres

Exemple : 


## L'apprentissage via Trainer


### LightningCLI
tout automatiser, les hyperparametres dans des fichiers de configuration, ou des commandes avec un help et tout ce qu'il faut pour utiliser sur les serveurs

Appel depuis le shell, il vaut mieux faire comme ça et surtout pas toucher au code pour ne pas introduire des bugs

### checkpoints
reutiliser

### La gestion des données
LightninDataModule permet aussi d'avoir une sous classe qui structure proprement la partie donnée

- **prepare_data** : Permet de telecharger les données et de faire les splits. Que sur le processus principal si parallelisation. Exemple : découpage train test sur prepare 
- **setup :** initialisation de la base de donnée, appelé sur chaque processus ou chacun des noeuds. 
- **train_dataloader**