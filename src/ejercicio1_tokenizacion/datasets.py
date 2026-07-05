"""
datasets.py (Ejercicio 1)
==========================
Carga y preprocesamiento de los corpus Ancora y CoNLL2002, según lo pedido
en las actividades 1 y 2 del taller:

  - CoNLL2002: se usan directamente los splits originales train/validation/test.
  - Ancora: se descarga el corpus completo (Universal Dependencies) y se
    construye MANUALMENTE el split 70/15/15 (train/validation/test).
"""

import random
import urllib.request

import nltk

# --------------------------------------------------------------------
# ANCORA
# --------------------------------------------------------------------

URL_ANCORA_TRAIN = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Spanish-AnCora/master/es_ancora-ud-train.conllu"
)
URL_ANCORA_VAL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Spanish-AnCora/master/es_ancora-ud-dev.conllu"
)
URL_ANCORA_TEST = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Spanish-AnCora/master/es_ancora-ud-test.conllu"
)


def _descargar(url):
    """Descarga el contenido de una URL como texto UTF-8."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req).read().decode("utf-8")


def procesar_ancora(texto_conllu):
    """Convierte texto en formato CoNLL-U en una lista de oraciones,
    cada una representada como una lista de palabras (tokens)."""
    oraciones = []
    oracion_actual = []

    for linea in texto_conllu.splitlines():
        linea = linea.strip()
        if linea.startswith("#"):
            continue
        if not linea:
            if oracion_actual:
                oraciones.append(oracion_actual)
                oracion_actual = []
            continue

        partes = linea.split("\t")
        if len(partes) < 2:
            continue
        if "-" in partes[0] or "." in partes[0]:
            continue  # tokens multi-palabra o nodos vacíos de UD

        word = partes[1]
        oracion_actual.append(word)

    if oracion_actual:
        oraciones.append(oracion_actual)
    return oraciones


def split_manual(oraciones, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Construye manualmente los conjuntos train/validation/test a partir de
    UNA sola lista de oraciones, respetando la proporción 70/15/15 pedida en
    el enunciado. Se baraja con una semilla fija para reproducibilidad."""
    oraciones = oraciones.copy()
    rng = random.Random(seed)
    rng.shuffle(oraciones)

    n = len(oraciones)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = oraciones[:n_train]
    val = oraciones[n_train:n_train + n_val]
    test = oraciones[n_train + n_val:]
    return train, val, test


def cargar_ancora(train_ratio=0.70, val_ratio=0.15, seed=42):
    """Descarga y unifica los tres archivos oficiales de UD_Spanish-AnCora,
    y construye manualmente el split 70/15/15 solicitado por el enunciado
    (no se usan las particiones ya definidas por UD)."""
    print("\nCargando y preprocesando Ancora desde Universal Dependencies...")

    txt_ancora_completo = (
        _descargar(URL_ANCORA_TRAIN)
        + "\n"
        + _descargar(URL_ANCORA_VAL)
        + "\n"
        + _descargar(URL_ANCORA_TEST)
    )

    ancora_todas = procesar_ancora(txt_ancora_completo)
    print(f"✓ Ancora unificado: {len(ancora_todas)} oraciones totales.")

    ancora_train, ancora_val, ancora_test = split_manual(
        ancora_todas, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    print(
        f"✓ Split manual Ancora -> Train: {len(ancora_train)} "
        f"({len(ancora_train) / len(ancora_todas):.1%}), "
        f"Val: {len(ancora_val)} ({len(ancora_val) / len(ancora_todas):.1%}), "
        f"Test: {len(ancora_test)} ({len(ancora_test) / len(ancora_todas):.1%})"
    )
    return ancora_train, ancora_val, ancora_test


# --------------------------------------------------------------------
# CoNLL2002
# --------------------------------------------------------------------

def cargar_conll2002():
    """Carga los splits originales train/validation/test de CoNLL2002
    (español) vía NLTK. `esp.train` / `esp.testa` / `esp.testb`
    corresponden respectivamente a train / validation / test."""
    print("\nCargando y preprocesando CoNLL2002 vía NLTK...")
    nltk.download("conll2002", quiet=True)
    from nltk.corpus import conll2002

    conll_train = [[tok[0] for tok in sent] for sent in conll2002.iob_sents("esp.train")]
    conll_val = [[tok[0] for tok in sent] for sent in conll2002.iob_sents("esp.testa")]
    conll_test = [[tok[0] for tok in sent] for sent in conll2002.iob_sents("esp.testb")]
    print("✓ CoNLL2002 procesado con éxito.")
    return conll_train, conll_val, conll_test


def cargar_todos_los_corpus(train_ratio=0.70, val_ratio=0.15, seed=42):
    """Carga y prepara ambos corpus (Ancora + CoNLL2002) y muestra un
    resumen final de tamaños."""
    ancora_train, ancora_val, ancora_test = cargar_ancora(train_ratio, val_ratio, seed)
    conll_train, conll_val, conll_test = cargar_conll2002()

    print("\nResumen final de estructuras cargadas:")
    print(f" - Ancora    -> Train: {len(ancora_train)}, Val: {len(ancora_val)}, Test: {len(ancora_test)} oraciones.")
    print(f" - CoNLL2002 -> Train: {len(conll_train)}, Val: {len(conll_val)}, Test: {len(conll_test)} oraciones.")

    return {
        "ancora": {"train": ancora_train, "validation": ancora_val, "test": ancora_test},
        "conll2002": {"train": conll_train, "validation": conll_val, "test": conll_test},
    }
