"""
visualizacion.py (Ejercicio 2)
================================
Visualización de relaciones semánticas con PCA y t-SNE (actividad 4),
para distintos tamaños de subconjuntos de palabras (100, 5000, 10000), y
comparación cuantitativa de ambos algoritmos en términos de estructura
local y global, tal como exige el enunciado.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from scipy.spatial.distance import pdist


def visualizar_embeddings(modelo, nombre_modelo, n_palabras, out_dir):
    """Genera y guarda visualizaciones PCA y t-SNE de las `n_palabras` más
    frecuentes del vocabulario de `modelo` (Word2Vec o FastText), y calcula
    métricas de fidelidad de cada proyección respecto al espacio original."""
    os.makedirs(out_dir, exist_ok=True)
    vocabulario = list(modelo.wv.key_to_index.keys())
    n_reales = min(n_palabras, len(vocabulario))

    if n_reales < 3:
        print(f"Vocabulario insuficiente para visualizar {nombre_modelo}.")
        return None

    palabras_muestra = vocabulario[:n_reales]
    vectores = np.array([modelo.wv[p] for p in palabras_muestra])

    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(vectores)
    ruta_pca = _graficar(
        coords_pca, palabras_muestra, n_reales,
        titulo=f"PCA - {nombre_modelo} ({n_reales} palabras)",
        xlabel="Componente 1", ylabel="Componente 2",
        ruta=os.path.join(out_dir, f"pca_{nombre_modelo}_{n_reales}.png"),
    )

    perplexity = min(30, max(5, n_reales // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                init="pca", learning_rate="auto")
    coords_tsne = tsne.fit_transform(vectores)
    ruta_tsne = _graficar(
        coords_tsne, palabras_muestra, n_reales,
        titulo=f"t-SNE - {nombre_modelo} ({n_reales} palabras)",
        xlabel="Dimensión 1", ylabel="Dimensión 2",
        ruta=os.path.join(out_dir, f"tsne_{nombre_modelo}_{n_reales}.png"),
    )

    metricas = _comparar_estructura_global_local(
        vectores, coords_pca, coords_tsne, nombre_modelo, n_reales, out_dir
    )

    return {"pca": ruta_pca, "tsne": ruta_tsne, "metricas_comparacion": metricas}


def _graficar(coords, palabras_muestra, n_reales, titulo, xlabel, ylabel, ruta):
    plt.figure(figsize=(9, 7))
    plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)

    for i in range(min(30, n_reales)):
        plt.annotate(palabras_muestra[i], (coords[i, 0], coords[i, 1]), fontsize=7)

    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Gráfica guardada en: {ruta}")
    return ruta


def _comparar_estructura_global_local(vectores, coords_pca, coords_tsne, nombre_modelo, n_reales, out_dir):
    """Calcula dos métricas cuantitativas para comparar PCA vs. t-SNE:

    - Estructura LOCAL: `trustworthiness` (sklearn) mide qué tan bien se
      preservan los vecinos más cercanos de cada punto al proyectar a 2D
      (1.0 = preservación perfecta de vecindarios locales).
    - Estructura GLOBAL: correlación de Pearson entre las distancias
      pareadas en el espacio original (300D) y en la proyección 2D
      (1.0 = las distancias relativas entre TODOS los puntos, no solo los
      vecinos, se conservan perfectamente).

    Ambas métricas se calculan para PCA y para t-SNE, y el resultado
    (impreso y guardado en .txt) fundamenta la conclusión pedida en el
    enunciado sobre el comportamiento de ambos algoritmos.
    """
    # trustworthiness exige n_neighbors < n_muestras / 2
    n_neighbors = max(1, min(5, (n_reales // 2) - 1))

    trust_pca = trustworthiness(vectores, coords_pca, n_neighbors=n_neighbors)
    trust_tsne = trustworthiness(vectores, coords_tsne, n_neighbors=n_neighbors)

    # Para distancias globales, se limita a una muestra si n_reales es grande,
    # para no calcular una matriz de distancias excesivamente costosa.
    idx_muestra = np.arange(n_reales) if n_reales <= 2000 else np.random.RandomState(42).choice(n_reales, 2000, replace=False)

    dist_orig = pdist(vectores[idx_muestra])
    dist_pca = pdist(coords_pca[idx_muestra])
    dist_tsne = pdist(coords_tsne[idx_muestra])

    corr_pca_global = float(np.corrcoef(dist_orig, dist_pca)[0, 1])
    corr_tsne_global = float(np.corrcoef(dist_orig, dist_tsne)[0, 1])

    mejor_local = "t-SNE" if trust_tsne > trust_pca else "PCA"
    mejor_global = "PCA" if corr_pca_global > corr_tsne_global else "t-SNE"

    texto = f"""\
Comparación PCA vs. t-SNE — Modelo: {nombre_modelo} — n={n_reales} palabras
{'=' * 70}
Estructura LOCAL (trustworthiness, k={n_neighbors} vecinos, 1.0 = ideal):
  PCA    : {trust_pca:.4f}
  t-SNE  : {trust_tsne:.4f}
  -> Mejor preservación local: {mejor_local}

Estructura GLOBAL (correlación de Pearson entre distancias pareadas
originales (300D) y proyectadas, 1.0 = ideal):
  PCA    : {corr_pca_global:.4f}
  t-SNE  : {corr_tsne_global:.4f}
  -> Mejor preservación global: {mejor_global}

Interpretación: PCA es una proyección lineal que prioriza la varianza
global de los datos, por lo que tiende a conservar mejor las distancias
relativas entre TODAS las palabras (estructura global), aunque a costa de
mezclar vecindarios locales cuando la varianza útil no cabe en solo 2
componentes. t-SNE, en cambio, optimiza explícitamente la preservación de
vecinos cercanos (estructura local), lo que suele producir agrupamientos
(clusters) visualmente más nítidos de palabras semánticamente
relacionadas, pero puede distorsionar o exagerar las distancias entre
grupos lejanos, por lo que no debe interpretarse la distancia global entre
clusters de t-SNE de forma literal.
"""
    print("\n" + texto)

    os.makedirs(out_dir, exist_ok=True)
    ruta_txt = os.path.join(out_dir, f"comparacion_pca_tsne_{nombre_modelo}_{n_reales}.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"✓ Comparación cuantitativa guardada en {ruta_txt}")

    return {
        "trustworthiness_pca": trust_pca,
        "trustworthiness_tsne": trust_tsne,
        "correlacion_global_pca": corr_pca_global,
        "correlacion_global_tsne": corr_tsne_global,
        "ruta_txt": ruta_txt,
    }
