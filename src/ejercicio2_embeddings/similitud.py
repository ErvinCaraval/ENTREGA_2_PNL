"""
similitud.py (Ejercicio 2)
============================
Análisis de similitud semántica y manejo de palabras OOV (actividad 5).
"""

import os

EXPLICACION_OOV = """
EXPLICACIÓN SOBRE PALABRAS OOV

Word2Vec aprende un vector independiente para cada palabra completa que aparece
en el corpus de entrenamiento. Si una palabra no apareció en el corpus o apareció
menos veces que el valor definido en min_count, Word2Vec no tendrá un vector para
ella y no podrá calcular palabras similares.

FastText funciona de manera diferente. En lugar de representar una palabra solo
como una unidad completa, la divide en fragmentos de caracteres llamados n-gramas.
Por ejemplo, una palabra como "ciberseguridad" puede dividirse en fragmentos como
"cib", "iber", "segur", "uridad" y otros. Aunque la palabra completa no haya sido
vista durante el entrenamiento, algunos de esos fragmentos pueden haber aparecido
en otras palabras.

Por esa razón, FastText puede construir una representación aproximada de palabras
fuera del vocabulario, conocidas como OOV. Esto es especialmente útil en español,
donde existen muchas variaciones de género, número, conjugaciones verbales,
prefijos, sufijos y palabras nuevas.
"""


def analizar_similitud(modelo_w2v, modelo_ft, palabras_vocab, palabras_oov, topn=10):
    """Compara las `topn` palabras más similares en Word2Vec y FastText para
    cada palabra de `palabras_vocab`, y demuestra el manejo de palabras OOV
    usando únicamente FastText (gracias a su representación por n-gramas de
    caracteres, que sí puede aproximar palabras nunca vistas)."""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE SIMILITUD SEMÁNTICA")
    print("=" * 80)

    resultados = {}

    for palabra in palabras_vocab:
        print(f"\nPalabra: {palabra}")
        resultados[palabra] = {}

        if palabra in modelo_w2v.wv.key_to_index:
            similares_w2v = modelo_w2v.wv.most_similar(palabra, topn=topn)
            print("\nWord2Vec:")
            print(similares_w2v)
            resultados[palabra]["word2vec"] = similares_w2v
        else:
            print("\nWord2Vec: palabra no encontrada.")
            resultados[palabra]["word2vec"] = None

        if palabra in modelo_ft.wv.key_to_index:
            similares_ft = modelo_ft.wv.most_similar(palabra, topn=topn)
            print("\nFastText:")
            print(similares_ft)
            resultados[palabra]["fasttext"] = similares_ft
        else:
            print("\nFastText: palabra no encontrada en el vocabulario.")
            resultados[palabra]["fasttext"] = None

    print("\n" + "-" * 80)
    print("PALABRAS OOV: solo FastText")
    print("-" * 80)

    for palabra in palabras_oov:
        print(f"\nPalabra OOV: {palabra}")
        try:
            similares_oov = modelo_ft.wv.most_similar(palabra, topn=topn)
            print(similares_oov)
            resultados[palabra] = {"fasttext_oov": similares_oov}
        except KeyError:
            print("No fue posible calcular similitud.")
            resultados[palabra] = {"fasttext_oov": None}

    return resultados


def guardar_explicacion_oov(output_dir):
    """Guarda en disco la explicación de por qué FastText sí puede tratar
    palabras OOV y Word2Vec no."""
    os.makedirs(output_dir, exist_ok=True)
    ruta = os.path.join(output_dir, "ejercicio2_explicacion_oov.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(EXPLICACION_OOV)
    print(f"\n✓ Explicación OOV guardada en {ruta}")
    return ruta
