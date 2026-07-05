"""
utils.py
========
Funciones auxiliares de propósito general, reutilizadas por varios de los
tres ejercicios del taller.
"""


def display_first_sentences(sentences, n, nombre_dataset):
    """Muestra en consola las primeras `n` oraciones de un conjunto de datos.

    Parameters
    ----------
    sentences : list[list[str]]
        Lista de oraciones, cada una representada como lista de palabras.
    n : int
        Número de oraciones a mostrar.
    nombre_dataset : str
        Nombre descriptivo del conjunto (para el encabezado impreso).
    """
    print(f"\n--- Primeras {n} oraciones originales de {nombre_dataset} ---")
    for i in range(min(n, len(sentences))):
        print(f"Oración {i + 1}: " + " ".join(sentences[i]))
