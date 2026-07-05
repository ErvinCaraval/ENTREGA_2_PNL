# Entrega_2_PNL — Taller #1: Datasets, tokenización y embeddings

Proyecto modularizado del Taller #1 (PLN), organizado en tres ejercicios
independientes, cada uno con su propio subpaquete Python dentro de `src/`.

## Estructura del proyecto

```
Entrega_2_PNL/
├── main.py                          # Punto de entrada único (CLI o import)
├── requirements.txt                 # Dependencias del proyecto
├── README.md
├── pdfs_taller/                     # <- Coloca aquí tus PDFs para el Ejercicio 3
├── resultados/                      # Salidas generadas (CSV, .npy, .png, .txt)
│   ├── ejercicio1_tokenizacion/
│   ├── ejercicio2_embeddings/
│   └── ejercicio3_pdf_embeddings/
├── notebooks/
│   └── Taller1_Main.ipynb           # Notebook orquestador (Colab/Drive)
└── src/
    ├── config.py                    # Rutas y semillas globales
    ├── utils.py                     # Utilidades compartidas
    ├── ejercicio1_tokenizacion/
    │   ├── datasets.py              # Carga Ancora + CoNLL2002, split 70/15/15
    │   ├── tokenizadores.py         # WordPiece (BETO) y SentencePiece (t5-small)
    │   ├── comparacion.py           # Comparación tabular + conclusión (punto 7)
    │   └── pipeline.py              # Orquesta actividades 1-7
    ├── ejercicio2_embeddings/
    │   ├── corpus.py                # Carga streaming + limpieza del corpus
    │   ├── entrenamiento.py         # Word2Vec y FastText
    │   ├── visualizacion.py         # PCA y t-SNE
    │   ├── similitud.py             # Similitud semántica + palabras OOV
    │   └── pipeline.py              # Orquesta actividades 1-5
    └── ejercicio3_pdf_embeddings/
        ├── carga_pdf.py             # Carga de PDFs con pymupdf4llm
        ├── chunking.py              # RecursiveCharacterTextSplitter
        ├── embeddings.py            # 4 modelos de Sentence-Transformers
        ├── busqueda.py              # Similitud coseno consulta vs. fragmentos
        ├── visualizacion.py         # PCA consulta vs. fragmento recuperado
        └── pipeline.py              # Orquesta actividades 1-5
```

## Cómo usarlo en Google Drive + Google Colab

1. Sube toda la carpeta `Entrega_2_PNL` (tal cual, con su estructura) a tu
   Google Drive, por ejemplo a `MiUnidad/Entrega_2_PNL`.
2. Coloca los PDFs que quieras analizar en el Ejercicio 3 dentro de
   `Entrega_2_PNL/pdfs_taller/`.
3. Abre `notebooks/Taller1_Main.ipynb` desde Drive con Google Colab
   (clic derecho -> "Abrir con" -> "Google Colaboratory").
4. Ejecuta las celdas en orden. La primera celda:
   - Monta tu Google Drive (`drive.mount('/content/drive')`).
   - Instala las dependencias (`requirements.txt`).
   - Ajusta `config.BASE_DIR` a la ruta de la carpeta en tu Drive y agrega
     el proyecto al `sys.path` para poder hacer `from src... import ...`.
5. Las siguientes celdas ejecutan cada ejercicio de forma independiente
   (Ejercicio 1, luego Ejercicio 2, luego Ejercicio 3), imprimiendo en el
   notebook todo el detalle pedido en el enunciado (ejemplos originales,
   tokens, comparaciones, gráficas, similitudes, conclusiones) y guardando
   los artefactos (CSV, modelos `.model`, embeddings `.npy`, gráficas
   `.png`, conclusiones `.txt`) en `resultados/` **dentro de tu Drive**, de
   modo que persisten entre sesiones.

## Cómo usarlo localmente (fuera de Colab)

### 1) Crear y activar un entorno virtual

#### Linux / macOS

```bash
cd Entrega_2_PNL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
cd Entrega_2_PNL
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación por política de ejecución, usa:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Si prefieres usar el símbolo del sistema de Windows (cmd), la activación es:

```bat
.venv\Scripts\activate.bat
```

Para salir del entorno virtual en cualquier sistema:

```bash
deactivate
```

### 2) Ejecutar el proyecto

```bash
# Ejecutar todo el taller:
python main.py --ejercicios 1 2 3

# Ejecutar solo un ejercicio, por ejemplo el 2, con un corpus más pequeño:
python main.py --ejercicios 2 --n-sentencias-ej2 5000

# Usar una carpeta base distinta (por ejemplo un Drive montado con rclone):
python main.py --base-dir /ruta/a/Entrega_2_PNL
```

También puedes importar cada pipeline por separado desde Python:

```python
from src import config
from src.ejercicio1_tokenizacion.pipeline import ejecutar_ejercicio1

resultado_ej1 = ejecutar_ejercicio1(config.OUTPUT_DIR_EJ1)
```

## Cobertura de los puntos del enunciado

| Punto del PDF | Módulo(s) responsable(s) |
|---|---|
| Ej. 1 — Cargar Conll2002 y Ancora | `ejercicio1_tokenizacion/datasets.py` |
| Ej. 1 — Split manual 70/15/15 de Ancora | `datasets.split_manual` |
| Ej. 1 — Mostrar ejemplos originales | `utils.display_first_sentences` (vía `pipeline.py`) |
| Ej. 1 — Tokenización WordPiece (BETO) | `tokenizadores.tokenizar_con_beto` |
| Ej. 1 — Tokenización SentencePiece (t5-small) | `tokenizadores.tokenizar_con_sentencepiece` |
| Ej. 1 — Comparación WordPiece vs. SentencePiece | `comparacion.comparar_tokenizadores` |
| Ej. 1 — Conclusión (punto 7) | `comparacion.generar_conclusion_ejercicio1` |
| Ej. 2 — Carga streaming de corpus en español | `ejercicio2_embeddings/corpus.py` |
| Ej. 2 — Preprocesamiento (limpieza, stopwords) | `corpus.limpiar_oracion` |
| Ej. 2 — Entrenar Word2Vec y FastText | `ejercicio2_embeddings/entrenamiento.py` |
| Ej. 2 — Visualización PCA / t-SNE (100/5000/10000) | `ejercicio2_embeddings/visualizacion.py` |
| Ej. 2 — Comparación cuantitativa PCA vs. t-SNE (estructura global/local) | `visualizacion._comparar_estructura_global_local` (trustworthiness + correlación de distancias) |
| Ej. 2 — Similitud semántica + OOV con FastText | `ejercicio2_embeddings/similitud.py` |
| Ej. 3 — Carga de PDFs (LangChain + pymupdf4llm) | `ejercicio3_pdf_embeddings/carga_pdf.py` |
| Ej. 3 — Chunking (RecursiveCharacterTextSplitter) | `ejercicio3_pdf_embeddings/chunking.py` |
| Ej. 3 — Embeddings con 4 modelos Sentence-Transformers | `ejercicio3_pdf_embeddings/embeddings.py` |
| Ej. 3 — Consulta + similitud coseno | `ejercicio3_pdf_embeddings/busqueda.py` |
| Ej. 3 — Visualización PCA **y t-SNE** consulta vs. fragmento | `ejercicio3_pdf_embeddings/visualizacion.py` |

## Desviaciones conscientes respecto al enunciado (y por qué)

- **CoNLL2002**: el enunciado ilustra la carga con
  `dataset["train"][0]["tokens"]` (librería `datasets` de HuggingFace).
  El dataset `conll2002` fue retirado del Hub de HuggingFace, por lo que
  se carga mediante `nltk.corpus.conll2002` (`esp.train` / `esp.testa` /
  `esp.testb`), que son exactamente los mismos splits (train/validation/test)
  y el mismo contenido, solo con una fuente distinta. El resultado y el
  formato de tokens es idéntico al esperado.
- **`spanish_billion_words`**: ver nota abajo — se sustituye por
  `allenai/c4` (config `"es"`) en streaming, porque el script de carga
  original del dataset ya no funciona con versiones recientes de `datasets`.
- **Ejercicio 2, "guardar embeddings intermedios"**: el enunciado permite
  procesar por lotes y guardar `.npy` si la memoria es limitada. Gensim no
  expone embeddings parciales durante el entrenamiento, así que lo que se
  guarda por lotes son las oraciones ya limpias (entrada al entrenamiento),
  cumpliendo el mismo propósito (no perder el avance si se interrumpe la
  sesión) sin bloquear la ejecución.

## Notas importantes

- **`spanish_billion_words`**: este dataset ya no carga con versiones
  recientes de `datasets` (depende de un script remoto obsoleto). Se usa
  como alternativa `allenai/c4` (config `"es"`) en modo `streaming=True`,
  cumpliendo el mismo requisito del enunciado (corpus grande en español,
  cargado sin saturar memoria). Puedes cambiar el dataset editando los
  parámetros `dataset_name` / `dataset_config` de
  `ejecutar_ejercicio2(...)`.
- Todas las semillas aleatorias están fijadas en `config.SEED = 42` para
  que los resultados (splits, modelos, t-SNE) sean reproducibles.
- El Ejercicio 3 requiere que existan archivos `.pdf` en `pdfs_taller/`
  antes de ejecutarlo; si la carpeta está vacía, el pipeline lo indica
  claramente y no falla el resto del notebook.
