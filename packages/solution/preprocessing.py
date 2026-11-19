from typing import Any
import cv2
import numpy as np

# Intervallo HSV calibrato per il giallo/arancione dei Duckiebot
LOWER_HSV = np.array([10, 110, 100])
UPPER_HSV = np.array([35, 255, 255])


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """
    Converte l'immagine RGB in una maschera normalizzata [0, 1]
    che evidenzia i Duckiebot e ignora la parte alta (orizzonte).
    """
    h, w, _ = image_rgb.shape

    # 1. Conversione da RGB a HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # 2. Filtraggio del colore giallo/arancione
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    # 3. Pulizia del rumore (opzionale ma consigliata)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 4. Normalizzazione tra 0.0 e 1.0
    mask_float = mask.astype(np.float32) / 255.0

    # 5. Ignora il 40% superiore dell'immagine (soffitto, luci, ostacoli lontani)
    mask_float[: int(h * 0.4), :] = 0.0

    return mask_float