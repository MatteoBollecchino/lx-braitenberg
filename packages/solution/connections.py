from typing import Tuple
import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Matrice dei pesi per il motore SINISTRO.
    Ostacolo a sinistra -> peso positivo (accelera per sterzare a destra).
    Ostacolo a destra -> peso negativo (rallenta per sterzare a sinistra).
    """
    h, w = shape
    res = np.zeros(shape=shape, dtype="float32")

    # Gradiente verticale: 0 in alto, 1 in basso (reagisce di più agli ostacoli vicini)
    vertical_gradient = np.linspace(0.0, 1.0, h, dtype="float32")[:, None]

    # Metà sinistra: spinge in avanti (+1) | Metà destra: frena (-1)
    res[:, : w // 2] = 1.0
    res[:, w // 2 :] = -1.0

    return res * vertical_gradient


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Matrice dei pesi per il motore DESTRO.
    Speculare rispetto al motore sinistro.
    """
    h, w = shape
    res = np.zeros(shape=shape, dtype="float32")

    vertical_gradient = np.linspace(0.0, 1.0, h, dtype="float32")[:, None]

    # Metà sinistra: frena (-1) | Metà destra: spinge in avanti (+1)
    res[:, : w // 2] = -1.0
    res[:, w // 2 :] = 1.0

    return res * vertical_gradient