"""
visualizacion.py (Ejercicio 3)
================================
Visualización en PCA y t-SNE de la consulta vs. el fragmento recuperado,
para cada modelo de embeddings (actividad 5). El enunciado exige
explícitamente ambas técnicas ("Visualización en PCA y Tsne").
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualizar_consulta_vs_fragmento(nombre_modelo, emb_consulta, vectores_todos, idx_mejor, out_dir):
    """Genera dos gráficas 2D (PCA y t-SNE) mostrando la consulta en rojo y
    el fragmento recuperado en verde, en el contexto de todos los demás
    fragmentos (en gris) para dar referencia espacial. Cada gráfica incluye
    título (con el nombre del modelo), leyenda y etiquetas de ejes."""
    os.makedirs(out_dir, exist_ok=True)

    todos = np.vstack([vectores_todos, emb_consulta.reshape(1, -1)])
    n_puntos = todos.shape[0]

    ruta_pca = _graficar_proyeccion(
        metodo=PCA(n_components=2, random_state=42),
        todos=todos, idx_mejor=idx_mejor,
        titulo=f"Consulta vs. fragmento recuperado — PCA — Modelo: {nombre_modelo}",
        xlabel="Componente 1", ylabel="Componente 2",
        ruta=os.path.join(out_dir, f"pca_consulta_{nombre_modelo}.png"),
    )

    perplexity = min(30, max(5, (n_puntos - 1) // 3))
    ruta_tsne = _graficar_proyeccion(
        metodo=TSNE(n_components=2, random_state=42, perplexity=perplexity,
                     init="pca", learning_rate="auto"),
        todos=todos, idx_mejor=idx_mejor,
        titulo=f"Consulta vs. fragmento recuperado — t-SNE — Modelo: {nombre_modelo}",
        xlabel="Dimensión 1", ylabel="Dimensión 2",
        ruta=os.path.join(out_dir, f"tsne_consulta_{nombre_modelo}.png"),
    )

    return {"pca": ruta_pca, "tsne": ruta_tsne}


def _graficar_proyeccion(metodo, todos, idx_mejor, titulo, xlabel, ylabel, ruta):
    coords = metodo.fit_transform(todos)

    coords_fragmentos = coords[:-1]
    coord_consulta = coords[-1]
    coord_mejor = coords_fragmentos[idx_mejor]

    plt.figure(figsize=(8, 6))
    plt.scatter(coords_fragmentos[:, 0], coords_fragmentos[:, 1],
                s=15, alpha=0.3, color="gray", label="Otros fragmentos")
    plt.scatter(*coord_mejor, s=120, color="green", marker="^",
                label="Fragmento más similar", edgecolors="black", zorder=5)
    plt.scatter(*coord_consulta, s=150, color="red", marker="*",
                label="Consulta", edgecolors="black", zorder=5)

    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  ✓ Gráfica guardada -> {ruta}")
    return ruta
