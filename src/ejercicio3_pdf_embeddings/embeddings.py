"""
embeddings.py (Ejercicio 3)
=============================
Generación de embeddings de oraciones/fragmentos con cuatro modelos de
Sentence-Transformers (actividad 3), tal como exige el enunciado.
"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

MODELOS_EMBEDDING = {
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "paraphrase-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-m3": "BAAI/bge-m3",
    "paraphrase-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def generar_embeddings_por_modelo(chunks, output_dir, modelos=None):
    """Para cada modelo de `modelos` (por defecto, los 4 exigidos en el
    enunciado): carga el modelo, genera embeddings de todos los `chunks` y
    almacena tanto el modelo como los vectores resultantes."""
    modelos = modelos or MODELOS_EMBEDDING
    os.makedirs(output_dir, exist_ok=True)

    textos = [c.page_content for c in chunks]
    embeddings_por_modelo = {}

    for nombre, ruta_hf in modelos.items():
        print(f"\nCargando modelo de embeddings: {nombre} ({ruta_hf})...")
        modelo = SentenceTransformer(ruta_hf)
        vectores = modelo.encode(textos, show_progress_bar=True, normalize_embeddings=True)

        embeddings_por_modelo[nombre] = {
            "modelo": modelo,
            "vectores": np.array(vectores),
        }

        ruta_npy = os.path.join(output_dir, f"embeddings_{nombre}.npy")
        np.save(ruta_npy, np.array(vectores))
        print(f"✓ Embeddings generados para '{nombre}': shape={np.array(vectores).shape} -> {ruta_npy}")

    return embeddings_por_modelo
