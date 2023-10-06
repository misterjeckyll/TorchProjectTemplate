# Lancement avec la commande:
# lightning run app app_deploy_gradio.py
import pytorch_lightning as pl
from lightning.app.components import ServeGradio
import gradio as gr

from resnet_GTSRB import ResFour
from torchvision import transforms
import torch
import lightning as L
import os

class LitGradio(ServeGradio):
    inputs = gr.Image(type="pil", label="Upload Image for Object Detection")
    outputs = gr.Textbox(label='output')

    def __init__(self):
        super().__init__()
        self.ready = False
        self._model = self.build_model()
        self.labels = ["Limite de vitesse","Panneau STOP", "Panneau Danger", "Fin d'interdiction"]

    def predict(self, input_im) -> str:
        transform_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        input_im = transform_test(input_im).to(self.model.device)[None]
        y_pred = self._model(input_im) # prédiction du réseau
        # compute the most likely class
        label_id = torch.argmax(y_pred, dim=1).numpy()[0]
        label_text = self.labels[label_id]

        return label_text

    def build_model(self):
        model = ResFour().load_from_checkpoint("../checkpoints/resnetfour_epoch=001-val_loss=0.01.ckpt") # chargement du checkpoint
        model.eval() # passage du model en mode eval
        self.ready = True
        return model


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_gradio = LitGradio()

    def run(self):
        self.lit_gradio.run()

    def configure_layout(self):
        return [{"name": "home", "content": self.lit_gradio}]


app = L.LightningApp(RootFlow())