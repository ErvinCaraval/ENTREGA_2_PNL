"""
pipeline.py (Ejercicio 1)
==========================
Orquesta de principio a fin las actividades 1 a 7 del Ejercicio 1:
tokenización WordPiece vs. SentencePiece sobre Ancora y CoNLL2002.
"""

import os

import pandas as pd

from src.utils import display_first_sentences
from src.ejercicio1_tokenizacion.datasets import cargar_todos_los_corpus
from src.ejercicio1_tokenizacion.tokenizadores import (
    cargar_tokenizador_beto,
    cargar_tokenizador_t5,
    tokenizar_con_beto,
    tokenizar_con_sentencepiece,
)
from src.ejercicio1_tokenizacion.comparacion import (
    comparar_tokenizadores,
    generar_conclusion_ejercicio1,
)


def ejecutar_ejercicio1(output_dir, n_ejemplos=3, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Ejecuta el Ejercicio 1 completo y devuelve un diccionario con todos
    los artefactos generados (corpus, tokenizadores, métricas, conclusión)."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" EJERCICIO 1: TOKENIZACIÓN (WordPiece vs SentencePiece)")
    print("=" * 80)

    # Actividades 1 y 2: cargar y preprocesar los datasets
    corpus = cargar_todos_los_corpus(train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    conjuntos = [
        ("Ancora (Train)", corpus["ancora"]["train"]),
        ("Ancora (Validation)", corpus["ancora"]["validation"]),
        ("Ancora (Test)", corpus["ancora"]["test"]),
        ("CoNLL2002 (Train)", corpus["conll2002"]["train"]),
        ("CoNLL2002 (Validation)", corpus["conll2002"]["validation"]),
        ("CoNLL2002 (Test)", corpus["conll2002"]["test"]),
    ]

    # Actividad 3: mostrar ejemplos originales
    for nombre, conj in conjuntos:
        display_first_sentences(conj, n_ejemplos, nombre)

    # Actividades 4 y 5: cargar tokenizadores
    beto_tokenizer = cargar_tokenizador_beto()
    t5_tokenizer = cargar_tokenizador_t5()

    # Actividades 4, 5 y 6 combinadas sobre cada conjunto
    metricas_comparacion = []
    for nombre, conj in conjuntos:
        tokenizar_con_beto(conj, beto_tokenizer, nombre, n=n_ejemplos)
        tokenizar_con_sentencepiece(conj, t5_tokenizer, nombre, n=n_ejemplos)
        metricas_comparacion.extend(
            comparar_tokenizadores(conj, beto_tokenizer, t5_tokenizer, nombre, n=n_ejemplos)
        )

    df_comparacion = pd.DataFrame(metricas_comparacion)
    ruta_csv = os.path.join(output_dir, "ejercicio1_comparacion_tokenizadores.csv")
    df_comparacion.to_csv(ruta_csv, index=False)
    print(f"\n✓ Tabla comparativa guardada en {ruta_csv}")

    # Actividad 7: conclusión
    conclusion = generar_conclusion_ejercicio1(df_comparacion, output_dir)

    print("\n" + "#" * 80)
    print(" FIN DEL EJERCICIO 1")
    print("#" * 80)

    return {
        "corpus": corpus,
        "beto_tokenizer": beto_tokenizer,
        "t5_tokenizer": t5_tokenizer,
        "df_comparacion": df_comparacion,
        "conclusion": conclusion,
    }
