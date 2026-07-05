"""
pipeline.py (Ejercicio 2)
==========================
Orquesta de principio a fin las actividades 1 a 5 del Ejercicio 2:
Word2Vec y FastText sobre un corpus en español, con visualización
PCA/t-SNE y análisis de similitud + OOV.
"""

from src.ejercicio2_embeddings.corpus import cargar_y_preprocesar_corpus
from src.ejercicio2_embeddings.entrenamiento import entrenar_word2vec, entrenar_fasttext
from src.ejercicio2_embeddings.visualizacion import visualizar_embeddings
from src.ejercicio2_embeddings.similitud import analizar_similitud, guardar_explicacion_oov

PALABRAS_VOCABULARIO_DEFAULT = ["gobierno", "presidente", "educación", "trabajo", "casa"]
PALABRAS_OOV_DEFAULT = ["ciberseguridad", "teletrabajar"]


def ejecutar_ejercicio2(
    output_dir,
    n_sentencias=50_000,
    tamano_lote=10_000,
    tamanos_visualizacion=(100, 5000, 10000),
    palabras_vocab=None,
    palabras_oov=None,
    dataset_name="allenai/c4",
    dataset_config="es",
):
    """Ejecuta el Ejercicio 2 completo y devuelve un diccionario con todos
    los artefactos generados (corpus, modelos, resultados de similitud)."""
    palabras_vocab = palabras_vocab or PALABRAS_VOCABULARIO_DEFAULT
    palabras_oov = palabras_oov or PALABRAS_OOV_DEFAULT

    print("\n" + "=" * 80)
    print(" EJERCICIO 2: WORD2VEC Y FASTTEXT SOBRE CORPUS EN ESPAÑOL")
    print("=" * 80)

    # Actividades 1 y 2: cargar y preprocesar corpus
    oraciones = cargar_y_preprocesar_corpus(
        n_sentencias, tamano_lote=tamano_lote, out_dir=output_dir,
        dataset_name=dataset_name, dataset_config=dataset_config,
    )
    print("\nEjemplos de oraciones preprocesadas:")
    for oracion in oraciones[:3]:
        print(oracion)

    # Actividad 3: entrenar modelos
    modelo_w2v = entrenar_word2vec(oraciones, output_dir)
    modelo_ft = entrenar_fasttext(oraciones, output_dir)

    # Actividad 4: visualización PCA / t-SNE
    print("\nGenerando visualizaciones PCA y t-SNE...")
    rutas_visualizacion = {}
    for n in tamanos_visualizacion:
        rutas_visualizacion[f"word2vec_{n}"] = visualizar_embeddings(modelo_w2v, "Word2Vec", n, output_dir)
        rutas_visualizacion[f"fasttext_{n}"] = visualizar_embeddings(modelo_ft, "FastText", n, output_dir)

    # Actividad 5: similitud semántica + OOV
    resultados_similitud = analizar_similitud(modelo_w2v, modelo_ft, palabras_vocab, palabras_oov)
    guardar_explicacion_oov(output_dir)

    print("\n" + "=" * 80)
    print("FIN DEL EJERCICIO 2")
    print("=" * 80)
    print(f"\nResultados guardados en la carpeta: {output_dir}")

    return {
        "oraciones": oraciones,
        "modelo_w2v": modelo_w2v,
        "modelo_ft": modelo_ft,
        "rutas_visualizacion": rutas_visualizacion,
        "resultados_similitud": resultados_similitud,
    }
