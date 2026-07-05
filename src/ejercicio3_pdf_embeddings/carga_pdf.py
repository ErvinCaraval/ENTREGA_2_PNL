"""
carga_pdf.py (Ejercicio 3)
============================
Carga y preprocesamiento de documentos PDF (actividad 1), usando LangChain
junto con `pymupdf4llm` (recomendado por su mejor preservación del layout,
estructura y contenido textual).
"""

import glob
import os

import pymupdf4llm
from langchain_core.documents import Document


def cargar_pdfs(pdf_dir):
    """Busca y carga automáticamente todos los archivos PDF de `pdf_dir`,
    extrayendo su contenido textual (en Markdown) con pymupdf4llm."""
    if not os.path.exists(pdf_dir):
        print(f"\n⚠ El directorio '{pdf_dir}' no existe.")
        return []

    rutas_pdf = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"\nSe encontraron {len(rutas_pdf)} archivos PDF en '{pdf_dir}'.")

    documentos = []
    for ruta in rutas_pdf:
        print(f"  Cargando: {ruta}")
        contenido_markdown = pymupdf4llm.to_markdown(ruta)
        doc = Document(page_content=contenido_markdown, metadata={"source": ruta})
        documentos.append(doc)

    print(f"✓ Total de documentos cargados: {len(documentos)}")
    return documentos
