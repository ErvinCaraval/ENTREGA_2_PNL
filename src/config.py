"""
config.py
=========
Configuración global compartida por los tres ejercicios del Taller #1
(Datasets, tokenización y embeddings).

Toda ruta de salida se calcula a partir de BASE_DIR, de modo que basta con
cambiar esa variable (por ejemplo al path de la carpeta en Google Drive)
para que TODO el proyecto escriba sus resultados en el lugar correcto.
"""

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------
# Directorio raíz del proyecto.
#
# En Google Colab, tras montar el Drive, normalmente se hace:
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#   BASE_DIR = "/content/drive/MyDrive/Entrega_2_PNL"
#
# y luego:
#
#   from src import config
#   config.BASE_DIR = BASE_DIR
#   config.refrescar_rutas()
#
# Si se ejecuta localmente, el proyecto intenta detectarlo automáticamente
# a partir de la carpeta actual o del archivo de configuración.
# --------------------------------------------------------------------
def detectar_base_dir(base_dir=None):
    """Detecta la carpeta raíz del proyecto en Colab/Drive o localmente."""
    if base_dir is not None:
        return str(Path(base_dir).expanduser().resolve())

    env_base_dir = os.environ.get("PNL_BASE_DIR")
    if env_base_dir:
        return str(Path(env_base_dir).expanduser().resolve())

    if os.path.exists("/content/drive/MyDrive/Entrega_2_PNL"):
        return str(Path("/content/drive/MyDrive/Entrega_2_PNL").resolve())

    for inicio in (Path.cwd(), Path(__file__).resolve().parent.parent):
        for candidate in (inicio, *inicio.parents):
            if (candidate / "requirements.txt").exists() and (candidate / "src").exists():
                return str(candidate.resolve())

    return str(Path.cwd().resolve())


BASE_DIR = detectar_base_dir()

SEED = 42


def _rutas(base_dir):
    return {
        "BASE_DIR": base_dir,
        "OUTPUT_DIR": os.path.join(base_dir, "resultados"),
        "OUTPUT_DIR_EJ1": os.path.join(base_dir, "resultados", "ejercicio1_tokenizacion"),
        "OUTPUT_DIR_EJ2": os.path.join(base_dir, "resultados", "ejercicio2_embeddings"),
        "OUTPUT_DIR_EJ3": os.path.join(base_dir, "resultados", "ejercicio3_pdf_embeddings"),
        "PDF_DIR": os.path.join(base_dir, "pdfs_taller"),
    }


_r = _rutas(BASE_DIR)
OUTPUT_DIR = _r["OUTPUT_DIR"]
OUTPUT_DIR_EJ1 = _r["OUTPUT_DIR_EJ1"]
OUTPUT_DIR_EJ2 = _r["OUTPUT_DIR_EJ2"]
OUTPUT_DIR_EJ3 = _r["OUTPUT_DIR_EJ3"]
PDF_DIR = _r["PDF_DIR"]


def refrescar_rutas():
    """Recalcula todas las rutas de salida a partir del BASE_DIR actual.

    Debe llamarse después de modificar `config.BASE_DIR` (por ejemplo,
    tras montar Google Drive y apuntar a la carpeta `Entrega_2_PNL`)."""
    global BASE_DIR, OUTPUT_DIR, OUTPUT_DIR_EJ1, OUTPUT_DIR_EJ2, OUTPUT_DIR_EJ3, PDF_DIR
    BASE_DIR = detectar_base_dir(BASE_DIR)
    r = _rutas(BASE_DIR)
    OUTPUT_DIR = r["OUTPUT_DIR"]
    OUTPUT_DIR_EJ1 = r["OUTPUT_DIR_EJ1"]
    OUTPUT_DIR_EJ2 = r["OUTPUT_DIR_EJ2"]
    OUTPUT_DIR_EJ3 = r["OUTPUT_DIR_EJ3"]
    PDF_DIR = r["PDF_DIR"]
    crear_directorios()
    return r


def crear_directorios():
    """Crea todos los directorios de salida si no existen."""
    for d in (OUTPUT_DIR, OUTPUT_DIR_EJ1, OUTPUT_DIR_EJ2, OUTPUT_DIR_EJ3, PDF_DIR):
        os.makedirs(d, exist_ok=True)


def fijar_semillas(seed=SEED):
    """Fija las semillas de aleatoriedad para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)


crear_directorios()
