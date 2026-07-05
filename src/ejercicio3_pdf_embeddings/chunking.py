"""
chunking.py (Ejercicio 3)
===========================
Fragmentación (chunking) del texto extraído de los PDFs (actividad 2),
usando RecursiveCharacterTextSplitter con chunk_size=1000 y
chunk_overlap=200, tal como pide el enunciado.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def fragmentar_documentos(documentos, chunk_size=1000, chunk_overlap=200):
    """Fragmenta los documentos cargados en trozos de tamaño razonable.

    Justificación de parámetros:
      - chunk_size=1000: suficientemente grande para conservar contexto
        semántico completo (varios párrafos cortos o uno largo) sin exceder
        el límite de tokens típico de los modelos de embeddings evaluados
        (todos soportan >=512 tokens, y 1000 caracteres en español ronda
        ~180-220 tokens).
      - chunk_overlap=200: garantiza que las ideas que queden divididas justo
        en el borde de un fragmento (p. ej. una oración cortada a la mitad)
        sigan siendo recuperables desde el fragmento vecino, evitando
        pérdida de contexto en los bordes.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"✓ Documentos fragmentados en {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")
    return chunks
