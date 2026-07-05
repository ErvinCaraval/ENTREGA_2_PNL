"""
busqueda.py (Ejercicio 3)
===========================
Consulta de similitud semántica (actividad 4): para cada modelo, genera el
embedding de una consulta, calcula la similitud coseno contra todos los
fragmentos, e identifica el fragmento más similar.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def buscar_fragmento_mas_similar(consulta, chunks, embeddings_por_modelo):
    """Para cada modelo: genera el embedding de la consulta, calcula
    similitud coseno contra todos los fragmentos, e identifica el más
    similar."""
    resultados = {}
    for nombre, datos in embeddings_por_modelo.items():
        modelo = datos["modelo"]
        vectores = datos["vectores"]

        emb_consulta = modelo.encode([consulta], normalize_embeddings=True)
        similitudes = cosine_similarity(emb_consulta, vectores)[0]
        idx_mejor = int(np.argmax(similitudes))

        print(f"\n--- Modelo: {nombre} ---")
        print(f"Consulta: {consulta}")
        print(f"Fragmento más similar (similitud={similitudes[idx_mejor]:.4f}):")
        print(f"  {chunks[idx_mejor].page_content[:300]}...")

        resultados[nombre] = {
            "emb_consulta": emb_consulta[0],
            "idx_mejor": idx_mejor,
            "similitud": float(similitudes[idx_mejor]),
            "fragmento_texto": chunks[idx_mejor].page_content,
        }
    return resultados
