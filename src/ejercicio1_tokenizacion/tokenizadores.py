"""
tokenizadores.py (Ejercicio 1)
================================
Carga de los dos tokenizadores a comparar (actividades 4 y 5):

  - WordPiece del modelo BETO (dccuchile/bert-base-spanish-wwm-cased).
  - SentencePiece de t5-small (google-t5/t5-small), usando los prefijos "▁".
"""

from transformers import AutoTokenizer, T5Tokenizer


def cargar_tokenizador_beto():
    """Carga el tokenizador WordPiece de BETO."""
    print("\nCargando tokenizador WordPiece de BETO...")
    return AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")


def cargar_tokenizador_t5():
    """Carga el tokenizador SentencePiece de t5-small."""
    print("\nCargando tokenizador SentencePiece de t5-small...")
    return T5Tokenizer.from_pretrained("google-t5/t5-small")


def tokenizar_con_beto(sentencias, tokenizer, nombre_corpus, n=3):
    """Aplica WordPiece (BETO) a las primeras `n` oraciones del corpus."""
    print(f"\n{'=' * 70}")
    print(f"Corpus: {nombre_corpus}  |  Tokenizador: WordPiece (BETO)")
    print(f"{'=' * 70}")

    resultados = []
    for i in range(min(n, len(sentencias))):
        palabras = sentencias[i]
        print(f"\n[Oración {i + 1}] Original ({len(palabras)} palabras):")
        print("  " + " | ".join(palabras))

        subtokens_total = []
        piezas_por_palabra = []
        for palabra in palabras:
            sub = tokenizer.tokenize(palabra)
            piezas_por_palabra.append((palabra, sub))
            subtokens_total.extend(sub)

        splits = [(p, s) for p, s in piezas_por_palabra if len(s) > 1]
        if splits:
            print("  Palabras fragmentadas por el vocabulario:")
            for palabra, sub in splits:
                print(f"    '{palabra}' → {sub}")

        print(f"  Subtokens obtenidos: {subtokens_total}")
        print(f"  Resumen: {len(palabras)} palabras originales  →  {len(subtokens_total)} subtokens.")

        resultados.append({
            "corpus": nombre_corpus,
            "oracion_idx": i,
            "n_palabras": len(palabras),
            "n_subtokens": len(subtokens_total),
            "n_palabras_fragmentadas": len(splits),
        })
    return resultados


def tokenizar_con_sentencepiece(sentencias, tokenizer, nombre_corpus, n=3):
    """Aplica SentencePiece (T5) a las primeras `n` oraciones del corpus."""
    print(f"\n{'=' * 70}")
    print(f"Corpus: {nombre_corpus}  |  Tokenizador: SentencePiece (t5-small)")
    print(f"{'=' * 70}")

    resultados = []
    for i in range(min(n, len(sentencias))):
        palabras = sentencias[i]
        print(f"\n[Oración {i + 1}] Original ({len(palabras)} palabras):")
        print("  " + " | ".join(palabras))

        oracion_texto = " ".join(palabras)
        subtokens_total = tokenizer.tokenize(oracion_texto)

        print(f"  Subtokens obtenidos: {subtokens_total}")
        print(f"  Resumen: {len(palabras)} palabras originales  →  {len(subtokens_total)} subtokens.")

        resultados.append({
            "corpus": nombre_corpus,
            "oracion_idx": i,
            "n_palabras": len(palabras),
            "n_subtokens": len(subtokens_total),
        })
    return resultados
