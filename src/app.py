"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU
          you should choose device='cpu'
"""
from pathlib import Path
from typing import Dict, List, Optional

import torch
from biotransformers import BioTransformers
from deepchain.components import DeepChainApp
from torch import load

import joblib

Score = Dict[str, float]
ScoreList = List[Score]


class App(DeepChainApp):
    """DeepChain App template:

    - Implement score_names() and compute_score() methods.
    - Choose a a transformer available on BioTranfformers
    - Choose a personal keras/tensorflow model
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.transformer = BioTransformers(backend="protbert", device=device)
        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "family_model.pt"
        self._checkpoint_label_encoder: Optional[str] = "label_encoder.joblib"

        # load_model for tensorflow/keras model-load for pytorch model
        if self._checkpoint_filename is not None:
            self.model = load(self.get_checkpoint_path(__file__))

        # load the family label encoder
        self.label_encoder = joblib.load(self.get_checkpoint_label_encoder_path(__file__))

    def get_checkpoint_label_encoder_path(self, root_path: str) -> str:
        """
        Return solve checkpoint model path
        Args:
            root_path : path of the app file launch
        Raise:
            FileExistsError if no file are found inside the checkpoint folder
        """
        checkpoint_dir = (Path(root_path).parent / "../checkpoint").resolve()
        path_filename = checkpoint_dir / self._checkpoint_label_encoder
        if not path_filename.is_file():
            raise FileExistsError(
                f"File {self._checkpoint_label_encoder} not found in checkpoint folder."
                f" Set 'self._checkpoint_filename = None' if file not exists"
            )
        return path_filename

    @staticmethod
    def score_names() -> List[str]:
        """App Score Names. Must be specified.

        Example:
         return ["max_probability", "min_probability"]
        """
        return ["protein_family_id"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """Return a list of protein family id predictions for the given sequences.

        Args:
            sequences: sequences for which to predict family ids
        """

        try:
            # biotransformers v0.0.3
            x_embedding = self.transformer.compute_embeddings(sequences, pool_mode=["mean"])["mean"]
        except:
            # biotransformers v0.0.2
            x_embedding = self.transformer.compute_embeddings(sequences, pooling_list=["mean"])["mean"]

        y_hat = self.model(torch.tensor(x_embedding))
        predictions = torch.max(y_hat, dim=1)[1]
        predictions = predictions.detach().cpu().numpy()

        family_predictions = self.label_encoder.inverse_transform(predictions)
        family_list = [{"protein_family_id": family_pred} for family_pred in family_predictions]

        return family_list


if __name__ == "__main__":

    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]
    app = App("cpu")
    scores = app.compute_scores(sequences)
    print(scores)
