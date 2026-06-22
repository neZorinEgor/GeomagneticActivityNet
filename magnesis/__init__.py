from .core import GeomagneticNet
from .dataset import GeomagneticDataset
from .train import train_geomagnetic_model
from .utils import make_dataset

__version__ = "v0.1.0"
__all__ = [
    "GeomagneticNet",
    "GeomagneticDataset",
    "train_geomagnetic_model",
    "make_dataset",
]
