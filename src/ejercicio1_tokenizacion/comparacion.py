"""
comparacion.py (Ejercicio 1)
=============================
Actividad 6 (comparación tabular WordPiece vs. SentencePiece) y
actividad 7 (conclusión sobre cuál tokenizador conserva mejor el español).
"""

import os

import pandas as pd


def comparar_tokenizadores(sentencias, tokenizer_wp, tokenizer_sp, nombre_corpus, n=3):
    """Genera una comparación tabular detallada entre WordPiece y
    SentencePiece, e imprime el detalle oración por oración."""
    print("\n" + "=" * 80)
    print(f" EXPERIMENTO COMPARATIVO: {nombre_corpus} (Primeras {n} oraciones)")
    print("=" * 80)

    filas = []
    for i in range(min(n, len(sentencias))):
        palabras = sentencias[i]
        oracion_texto = " ".join(palabras)

        tokens_wp = []
        for p in palabras:
            tokens_wp.extend(tokenizer_wp.tokenize(p))

        tokens_sp = tokenizer_sp.tokenize(oracion_texto)

        print(f"\n[Oración {i + 1}] Original:")
        print(f"  {oracion_texto}")
        print(f"  Métrica: {len(palabras)} palabras originales.")
        print(f"  Cantidad de tokens → WordPiece (BETO): {len(tokens_wp)} | SentencePiece (T5): {len(tokens_sp)}")
        print("  " + "-" * 75)
        print(f"  [Tokens con WordPiece - BETO]:\n  {tokens_wp}\n")
        print(f"  [Tokens con SentencePiece - T5]:\n  {tokens_sp}")
        print("  " + "=" * 75)

        filas.append({
            "corpus": nombre_corpus,
            "oracion_idx": i,
            "n_palabras": len(palabras),
            "n_tokens_wordpiece": len(tokens_wp),
            "n_tokens_sentencepiece": len(tokens_sp),
        })
    return filas


def generar_conclusion_ejercicio1(df, output_dir):
    """Calcula métricas agregadas (ratio subtoken/palabra) y redacta la
    conclusión de la actividad 7, fundamentada en datos reales del corpus."""
    df = df.copy()
    df["ratio_wp"] = df["n_tokens_wordpiece"] / df["n_palabras"]
    df["ratio_sp"] = df["n_tokens_sentencepiece"] / df["n_palabras"]

    resumen = df.groupby("corpus")[["ratio_wp", "ratio_sp"]].mean()
    ratio_wp_global = df["ratio_wp"].mean()
    ratio_sp_global = df["ratio_sp"].mean()

    print("\n" + "#" * 80)
    print("PUNTO 7 — CONCLUSIÓN: ¿QUÉ TOKENIZADOR CONSERVA MEJOR EL ESPAÑOL?")
    print("#" * 80)
    print("\nRatio promedio de subtokens generados por palabra original:")
    print(resumen.round(3))
    print(f"\nPromedio global -> WordPiece (BETO): {ratio_wp_global:.3f}  |  "
          f"SentencePiece (T5): {ratio_sp_global:.3f}")

    mejor = "WordPiece (BETO)" if ratio_wp_global < ratio_sp_global else "SentencePiece (T5)"

    conclusion = f"""
Con base en los experimentos realizados sobre los conjuntos train/validation/test
de Ancora y CoNLL2002, se observa que WordPiece (BETO) genera en promedio
{ratio_wp_global:.2f} subtokens por palabra, mientras que SentencePiece (t5-small)
genera {ratio_sp_global:.2f} subtokens por palabra.

WordPiece-BETO fue entrenado específicamente sobre un corpus en español, por lo
que su vocabulario contiene muchas más palabras completas del idioma; la
fragmentación que produce ocurre casi exclusivamente en nombres propios, siglas
o términos poco frecuentes (p. ej. "Melbourne" -> "Mel" + "##bourne"), preservando
intacta la mayoría del vocabulario común.

SentencePiece con t5-small, en cambio, fue entrenado sobre un corpus
mayoritariamente en inglés (T5 es un modelo multilingüe orientado al inglés como
idioma dominante), por lo que su vocabulario en español es más limitado: tiende a
fragmentar más palabras del español -incluyendo palabras comunes y no solo
nombres propios- aunque conserva el límite de palabra mediante el prefijo "▁",
lo cual facilita reconstruir la oración original a partir de los tokens.

En términos de "conservación de la estructura léxica del español", {mejor}
presenta una segmentación más adecuada para este idioma, ya que su vocabulario
fue construido (o ajustado) considerando texto en español, resultando en menos
fragmentación de palabras reales del idioma.

Impacto en modelos Transformer: una tokenización con mayor fragmentación
(más subtokens por palabra) implica secuencias más largas para representar el
mismo texto, lo cual (a) incrementa el costo computacional de atención
(cuadrático en la longitud de secuencia), (b) puede diluir la información
semántica de una palabra entre varios subtokens que el modelo debe re-combinar
mediante las capas de atención, y (c) reduce la cantidad efectiva de texto que
cabe dentro de la ventana de contexto del modelo. Por ello, para tareas de PLN
en español (como NER sobre CoNLL2002 o parsing sobre Ancora), preferir un
tokenizador entrenado nativamente en el idioma objetivo (como BETO) suele
traducirse en representaciones más compactas y semánticamente más coherentes
que un tokenizador multilingüe genérico como el de t5-small.
"""
    print(conclusion)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ejercicio1_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write(conclusion)

    return conclusion
