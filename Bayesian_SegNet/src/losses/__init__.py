"""Loss functions for the project."""
from .categorical_aleatoric_loss import build_categorical_aleatoric_loss
from .categorical_crossentropy import build_categorical_crossentropy


# explicitly define the outward facing API of this package
__all__ = [
    build_categorical_aleatoric_loss.__name__,
    build_categorical_crossentropy.__name__,
]
