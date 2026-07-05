"""
corpus.py (Ejercicio 2)
========================
Carga y preprocesamiento del corpus en español para entrenar Word2Vec y
FastText (actividades 1 y 2).

Nota: el dataset "spanish_billion_words" ya no funciona directamente con
versiones recientes de la librería `datasets` porque depende de un script
remoto antiguo. Por eso se usa como alternativa `allenai/c4` (config "es"),
cargado en modo streaming para no saturar la memoria, tal como pide el
enunciado.
"""

import os
import re

import numpy as np
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS_ES = set(stopwords.words("spanish"))

_PUNTUACION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_NUMERO_RE = re.compile(r"\d+")


def limpiar_oracion(texto):
    """Convierte texto en una lista de palabras:
    - pasa a minúsculas;
    - elimina números;
    - elimina puntuación;
    - elimina stopwords;
    - conserva tildes y caracteres válidos del español."""
    texto = texto.lower()
    texto = _NUMERO_RE.sub(" ", texto)
    texto = _PUNTUACION_RE.sub(" ", texto)

    palabras = texto.split()

    palabras_limpias = [
        palabra
        for palabra in palabras
        if palabra not in STOPWORDS_ES and len(palabra) > 1
    ]

    return palabras_limpias


def cargar_y_preprocesar_corpus(
    n_sentencias,
    tamano_lote=10_000,
    out_dir=".",
    dataset_name="allenai/c4",
    dataset_config="es",
):
    """Carga un corpus en español mediante streaming (`load_dataset(...,
    streaming=True)`) y devuelve una lista de listas de palabras compatible
    con Gensim ([['raul', ...], ...]).

    También guarda lotes intermedios como archivos .npy (cada `tamano_lote`
    oraciones) para no perder el avance si se interrumpe la sesión."""
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"\nCargando corpus '{dataset_name}' ({dataset_config}) en modo "
        f"streaming (objetivo: {n_sentencias} oraciones)..."
    )

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
    )

    oraciones_limpias = []
    lote_actual = []
    numero_lote = 0

    for ejemplo in dataset:
        if len(oraciones_limpias) >= n_sentencias:
            break

        texto = ejemplo.get("text", "")

        for linea in texto.split("\n"):
            if len(oraciones_limpias) >= n_sentencias:
                break

            palabras = limpiar_oracion(linea)

            if len(palabras) >= 3:
                oraciones_limpias.append(palabras)
                lote_actual.append(palabras)

            if len(lote_actual) >= tamano_lote:
                _guardar_lote(lote_actual, numero_lote, out_dir)
                lote_actual = []
                numero_lote += 1

    if lote_actual:
        _guardar_lote(lote_actual, numero_lote, out_dir, final=True)

    print(f"Total de oraciones preprocesadas: {len(oraciones_limpias)}")
    return oraciones_limpias


def _guardar_lote(lote, numero_lote, out_dir, final=False):
    ruta_lote = os.path.join(out_dir, f"corpus_es_lote_{numero_lote}.npy")
    np.save(ruta_lote, np.array(lote, dtype=object), allow_pickle=True)
    etiqueta = "final " if final else ""
    print(f"Lote {etiqueta}{numero_lote} guardado: {len(lote)} oraciones")
