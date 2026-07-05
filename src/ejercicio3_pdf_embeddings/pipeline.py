"""
pipeline.py (Ejercicio 3)
==========================
Orquesta de principio a fin las actividades 1 a 5 del Ejercicio 3:
PDFs -> chunking -> embeddings (Sentence-Transformers) -> búsqueda
semántica -> visualización PCA y t-SNE.
"""

from src.ejercicio3_pdf_embeddings.carga_pdf import cargar_pdfs
from src.ejercicio3_pdf_embeddings.chunking import fragmentar_documentos
from src.ejercicio3_pdf_embeddings.embeddings import generar_embeddings_por_modelo, MODELOS_EMBEDDING
from src.ejercicio3_pdf_embeddings.busqueda import buscar_fragmento_mas_similar
from src.ejercicio3_pdf_embeddings.visualizacion import visualizar_consulta_vs_fragmento

CONSULTA_EJEMPLO_DEFAULT = "¿Cuál es el principio de precaución en materia ambiental?"


def ejecutar_ejercicio3(
    pdf_dir,
    output_dir,
    consulta=CONSULTA_EJEMPLO_DEFAULT,
    chunk_size=1000,
    chunk_overlap=200,
    modelos=None,
):
    """Ejecuta el Ejercicio 3 completo y devuelve un diccionario con todos
    los artefactos generados (chunks, embeddings, resultados de búsqueda,
    rutas de las gráficas)."""
    print("\n" + "=" * 80)
    print(" EJERCICIO 3: PDFs -> CHUNKING -> EMBEDDINGS (Sentence-Transformers)")
    print("=" * 80)

    # Actividad 1: carga de PDFs
    documentos = cargar_pdfs(pdf_dir)
    if not documentos:
        print(
            f"\n⚠ No se encontraron PDFs en '{pdf_dir}'. Coloca tus archivos "
            f".pdf allí y vuelve a ejecutar este bloque para completar el "
            f"Ejercicio 3."
        )
        return None

    # Actividad 2: chunking
    chunks = fragmentar_documentos(documentos, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Actividad 3: embeddings con 4 modelos de Sentence-Transformers
    embeddings_por_modelo = generar_embeddings_por_modelo(chunks, output_dir, modelos=modelos or MODELOS_EMBEDDING)

    # Actividad 4: consulta de similitud semántica
    resultados_busqueda = buscar_fragmento_mas_similar(consulta, chunks, embeddings_por_modelo)

    # Actividad 5: visualización PCA y t-SNE de consulta vs. fragmento recuperado
    print("\nGenerando visualizaciones PCA y t-SNE para cada modelo...")
    rutas_graficas = {}
    for nombre, datos in embeddings_por_modelo.items():
        res = resultados_busqueda[nombre]
        rutas_graficas[nombre] = visualizar_consulta_vs_fragmento(
            nombre, res["emb_consulta"], datos["vectores"], res["idx_mejor"], output_dir
        )

    print("\nInterpretación (completar en el informe con base en las gráficas):")
    for nombre, res in resultados_busqueda.items():
        print(f"  - {nombre}: similitud consulta-fragmento = {res['similitud']:.4f}")

    print("\n" + "#" * 80)
    print(" FIN DEL EJERCICIO 3")
    print("#" * 80)
    print(f"\nTodos los resultados (.npy, gráficas .png) se guardaron en: {output_dir}/")

    return {
        "documentos": documentos,
        "chunks": chunks,
        "embeddings_por_modelo": embeddings_por_modelo,
        "resultados_busqueda": resultados_busqueda,
        "rutas_graficas": rutas_graficas,
    }
