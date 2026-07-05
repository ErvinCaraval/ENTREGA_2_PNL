"""
main.py
=======
Punto de entrada único del proyecto "Entrega_2_PNL" (Taller #1: Datasets,
tokenización y embeddings).

Uso típico:

    python main.py --ejercicios 1 2 3

o, dentro de un notebook (Colab / Jupyter):

    from src import config
    config.BASE_DIR = "/content/drive/MyDrive/Entrega_2_PNL"
    config.refrescar_rutas()

    from main import ejecutar_todo
    resultados = ejecutar_todo(ejercicios=[1, 2, 3])

Cada ejercicio es independiente: pueden ejecutarse por separado, en el
orden que se desee, y cada uno escribe sus resultados en su propia
subcarpeta dentro de `resultados/`.
"""

import argparse

from src import config
from src.ejercicio1_tokenizacion.pipeline import ejecutar_ejercicio1
from src.ejercicio2_embeddings.pipeline import ejecutar_ejercicio2
from src.ejercicio3_pdf_embeddings.pipeline import ejecutar_ejercicio3


def ejecutar_todo(ejercicios=(1, 2, 3), n_sentencias_ej2=50_000, consulta_ej3=None):
    """Ejecuta los ejercicios indicados (1, 2 y/o 3) y devuelve un
    diccionario con los resultados de cada uno."""
    config.fijar_semillas()
    config.crear_directorios()

    resultados = {}

    if 1 in ejercicios:
        resultados["ejercicio1"] = ejecutar_ejercicio1(config.OUTPUT_DIR_EJ1)

    if 2 in ejercicios:
        resultados["ejercicio2"] = ejecutar_ejercicio2(
            config.OUTPUT_DIR_EJ2, n_sentencias=n_sentencias_ej2
        )

    if 3 in ejercicios:
        kwargs = {"pdf_dir": config.PDF_DIR, "output_dir": config.OUTPUT_DIR_EJ3}
        if consulta_ej3:
            kwargs["consulta"] = consulta_ej3
        resultados["ejercicio3"] = ejecutar_ejercicio3(**kwargs)

    return resultados


def _parsear_argumentos():
    parser = argparse.ArgumentParser(description="Taller #1 - Datasets, tokenización y embeddings")
    parser.add_argument(
        "--ejercicios", type=int, nargs="+", default=[1, 2, 3],
        help="Ejercicios a ejecutar: 1 (tokenización), 2 (Word2Vec/FastText), 3 (PDFs + Sentence-Transformers)",
    )
    parser.add_argument(
        "--base-dir", type=str, default=".",
        help="Carpeta raíz del proyecto (por ejemplo, la ruta a Entrega_2_PNL en Google Drive)",
    )
    parser.add_argument(
        "--n-sentencias-ej2", type=int, default=50_000,
        help="Número de oraciones a usar para entrenar Word2Vec/FastText en el Ejercicio 2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parsear_argumentos()
    config.BASE_DIR = args.base_dir
    config.refrescar_rutas()
    ejecutar_todo(ejercicios=args.ejercicios, n_sentencias_ej2=args.n_sentencias_ej2)
