from lightning.pytorch.cli import LightningCLI

from genie.dataset import LightningOpenX
from genie.model_visual_vq import VisualVQ_DINO_LAM


cli = LightningCLI(
    VisualVQ_DINO_LAM,
    LightningOpenX,
    seed_everything_default=42,
)
