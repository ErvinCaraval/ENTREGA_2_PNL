"""
entrenamiento.py (Ejercicio 2)
================================
Entrenamiento de los modelos Word2Vec y FastText (actividad 3), con la
configuración exigida en el enunciado: vector_size=300, window=5,
min_count=5, workers=4.
"""

import os

from gensim.models import Word2Vec, FastText


def entrenar_word2vec(oraciones, output_dir, vector_size=300, window=5,
                       min_count=5, workers=4, seed=42):
    """Entrena un modelo Word2Vec y lo guarda en `output_dir`."""
    print("\nEntrenando Word2Vec...")
    modelo = Word2Vec(
        sentences=oraciones,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )
    ruta = os.path.join(output_dir, "word2vec_es.model")
    modelo.save(ruta)
    print(f"Word2Vec entrenado. Vocabulario: {len(modelo.wv.key_to_index)} palabras. Guardado en {ruta}")
    return modelo


def entrenar_fasttext(oraciones, output_dir, vector_size=300, window=5,
                       min_count=5, workers=4, seed=42):
    """Entrena un modelo FastText y lo guarda en `output_dir`."""
    print("\nEntrenando FastText...")
    modelo = FastText(
        sentences=oraciones,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )
    ruta = os.path.join(output_dir, "fasttext_es.model")
    modelo.save(ruta)
    print(f"FastText entrenado. Vocabulario: {len(modelo.wv.key_to_index)} palabras. Guardado en {ruta}")
    return modelo
