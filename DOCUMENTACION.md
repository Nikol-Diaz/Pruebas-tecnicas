# Documentacion Completa: Clasificacion de Texto con BERT para Espanol

## Indice

1. [Conceptos Fundamentales de Machine Learning](#1-conceptos-fundamentales-de-machine-learning)
2. [Deep Learning y Redes Neuronales](#2-deep-learning-y-redes-neuronales)
3. [BERT y Transfer Learning](#3-bert-y-transfer-learning)
4. [El Pipeline Completo](#4-el-pipeline-completo)
5. [Codigo Celda por Celda](#5-codigo-celda-por-celda)

---

## 1. Conceptos Fundamentales de Machine Learning

### Que es Machine Learning?

Machine Learning (aprendizaje automatico) es una rama de la inteligencia artificial donde las computadoras **aprenden patrones a partir de datos** en lugar de ser programadas con reglas explicitas.

En nuestro caso: en lugar de escribir reglas como "si el texto contiene 'tarjeta', clasificar como 'Tarjetas'", el modelo aprende estas reglas automaticamente viendo miles de ejemplos.

### Tipos de Machine Learning

- **Aprendizaje supervisado** (lo que usamos aqui): el modelo recibe datos con sus respuestas correctas (etiquetas) y aprende a predecir la respuesta para datos nuevos.
- **Aprendizaje no supervisado**: el modelo busca patrones sin etiquetas (ej: clustering).
- **Aprendizaje por refuerzo**: el modelo aprende por prueba y error con recompensas.

### Clasificacion de Texto

Nuestro problema es de **clasificacion multi-clase**: dado un texto (descripcion del ticket), predecir a que area resolutora pertenece. Cada ticket pertenece a exactamente una clase.

### Conceptos Clave

#### Dataset (Conjunto de Datos)
El dataset es la coleccion de ejemplos que usamos. Cada ejemplo tiene:
- **Features (caracteristicas)**: el texto de la descripcion del ticket.
- **Label (etiqueta)**: el area resolutora correcta.

#### Train / Validation / Test Split
Dividimos los datos en 3 conjuntos:
- **Train (70%)**: el modelo aprende de estos datos.
- **Validation (15%)**: se usa durante el entrenamiento para monitorear si el modelo esta generalizando bien o memorizando (overfitting).
- **Test (15%)**: se usa UNA SOLA VEZ al final para la evaluacion definitiva. Nunca se usa durante el entrenamiento.

#### Epoch (Epoca)
Una epoca es una pasada completa por todos los datos de entrenamiento. Si tenemos 60,000 ejemplos de train, una epoca significa que el modelo vio los 60,000 una vez.

#### Batch (Lote)
Procesar todos los datos de una vez seria imposible por memoria. En su lugar, dividimos los datos en lotes (batches). Con batch_size=16, procesamos 16 ejemplos a la vez.

#### Loss (Perdida)
La loss es un numero que mide **que tan mal** predice el modelo. El objetivo del entrenamiento es minimizar la loss. Usamos **CrossEntropyLoss**, que es la funcion de perdida estandar para clasificacion multi-clase.

#### Learning Rate (Tasa de Aprendizaje)
El learning rate controla que tan grandes son los "pasos" que da el modelo al actualizar sus pesos:
- **Muy alto**: el modelo salta por todos lados y no converge.
- **Muy bajo**: el modelo aprende demasiado lento.
- Valor tipico para fine-tuning de BERT: **2e-5** (0.00002).

#### Overfitting vs Underfitting
- **Overfitting**: el modelo memoriza los datos de entrenamiento pero no generaliza a datos nuevos. Senal: train_loss baja pero val_loss sube.
- **Underfitting**: el modelo no aprende lo suficiente. Senal: ambas losses son altas.

#### Gradientes
Los gradientes son la base del aprendizaje. Indican en que direccion y cuanto ajustar cada parametro del modelo para reducir la loss. El algoritmo de **backpropagation** calcula estos gradientes automaticamente.

### Metricas de Evaluacion

#### Accuracy (Exactitud)
Porcentaje de predicciones correctas. Simple pero enganosa con datos desbalanceados. Si el 90% de los tickets son "Call Center", un modelo que siempre predice "Call Center" tendria 90% de accuracy pero seria inutil.

#### Precision
De todas las veces que el modelo predijo la clase X, cuantas realmente eran X.
- **Alta precision** = pocas falsas alarmas.
- Ejemplo: si predijo 100 tickets como "Sucursal" y 90 eran correctos → precision = 90%.

#### Recall (Sensibilidad)
De todos los tickets que realmente son de clase X, cuantos detecto el modelo.
- **Alto recall** = detecta casi todos los casos.
- Ejemplo: si habia 100 tickets de "Sucursal" y detecto 70 → recall = 70%.

#### F1-Score
Media armonica de precision y recall. Es la metrica mas equilibrada para datos desbalanceados.
- Formula: `2 * (precision * recall) / (precision + recall)`
- Ejemplo: precision=0.9, recall=0.1 → F1=0.18 (no 0.5 como la media aritmetica).

#### Macro vs Weighted
- **Macro**: calcula la metrica para cada clase y promedia sin importar el tamano de la clase. Trata todas las clases por igual.
- **Weighted**: promedia ponderando por el numero de ejemplos de cada clase. Las clases grandes pesan mas.

#### Top-K Accuracy
Verifica si la respuesta correcta esta entre las K predicciones mas probables. Util cuando el modelo no esta 100% seguro pero sugiere bien.

---

## 2. Deep Learning y Redes Neuronales

### Que es una Red Neuronal?

Una red neuronal es un modelo matematico inspirado en el cerebro humano. Consiste en capas de "neuronas" conectadas entre si, donde cada conexion tiene un peso (parametro) que se ajusta durante el entrenamiento.

### Componentes Clave

#### Parametros (Pesos)
Los parametros son los numeros que el modelo ajusta durante el entrenamiento. BERT tiene ~110 millones de parametros. Cada uno influye en como el modelo procesa el texto.

#### Capas Lineales (Dense/Linear)
Una capa lineal transforma un vector de dimension N a uno de dimension M multiplicando por una matriz de pesos y sumando un sesgo (bias): `output = input × W + b`.

#### Funciones de Activacion
Sin funciones de activacion, multiples capas lineales serian equivalentes a una sola. Las activaciones introducen no-linealidad:
- **ReLU**: `max(0, x)` — la mas comun, simple y efectiva.
- **Softmax**: convierte logits en probabilidades que suman 1. Se usa en la ultima capa para clasificacion.
- **Tanh**: similar a sigmoid pero con rango [-1, 1]. BERT la usa internamente.

#### Dropout
Tecnica de regularizacion que apaga aleatoriamente un porcentaje de neuronas durante el entrenamiento. Esto fuerza al modelo a no depender de neuronas especificas, previniendo overfitting.
- Dropout de 0.1 = apaga 10% de las neuronas aleatoriamente en cada paso.
- Solo se aplica durante entrenamiento, NO durante evaluacion.

#### Optimizer (Optimizador)
Algoritmo que actualiza los pesos del modelo usando los gradientes:
- **AdamW**: la variante mas usada para transformers. Combina momentum (memoria de pasos anteriores) con learning rate adaptativo por parametro, mas weight decay (regularizacion L2 corregida).

#### Scheduler
Ajusta el learning rate durante el entrenamiento:
- **Warmup**: sube gradualmente el LR de 0 al maximo en los primeros pasos. Esto estabiliza el inicio del entrenamiento.
- **Linear Decay**: despues del warmup, baja linealmente el LR hasta 0. Permite ajustes cada vez mas finos.

#### Gradient Clipping
Limita la magnitud de los gradientes para prevenir "exploding gradients" (gradientes enormes que desestabilizan el entrenamiento). Si la norma del gradiente excede un umbral (ej: 1.0), se escala proporcionalmente.

### Tensores

Un tensor es la estructura de datos fundamental en deep learning. Es una generalizacion de matrices a N dimensiones:
- Escalar (0D): un numero, ej: `5`
- Vector (1D): una lista, ej: `[1, 2, 3]`
- Matriz (2D): una tabla, ej: `[[1,2],[3,4]]`
- Tensor 3D+: ej: un batch de textos tokenizados tiene shape `[batch_size, max_length]`

### GPU vs CPU

Las GPUs (tarjetas graficas) son **masivamente paralelas**: pueden hacer miles de operaciones matematicas simultaneamente. Las redes neuronales requieren multiplicaciones de matrices enormes, que son ideales para GPUs.
- **CPU**: buena para tareas secuenciales y logica compleja.
- **GPU (CUDA)**: 10-50x mas rapida para deep learning.

---

## 3. BERT y Transfer Learning

### Que es BERT?

BERT (Bidirectional Encoder Representations from Transformers) es un modelo de lenguaje pre-entrenado por Google en 2018. Fue entrenado leyendo millones de textos y aprendiendo a entender el contexto de las palabras.

### Que es BETO?

BETO es la version de BERT entrenada especificamente con textos en espanol por la Universidad de Chile. Modelo: `dccuchile/bert-base-spanish-wwm-cased`.
- **spanish**: entrenado con textos en espanol.
- **wwm (Whole Word Masking)**: tecnica de pre-entrenamiento mejorada.
- **cased**: distingue mayusculas de minusculas.

### Arquitectura de BERT

BERT-base tiene:
- **12 capas (layers)** de encoder transformer.
- **768 dimensiones** en el vector oculto (hidden size).
- **12 cabezas de atencion** (attention heads).
- **~110 millones de parametros** en total.

#### Tokenizacion
BERT no trabaja con palabras completas sino con **sub-palabras (tokens)**. El tokenizador divide el texto en piezas que BERT conoce:
- Ejemplo: "autenticacion" → ["autent", "##icacion"]
- Tokens especiales: `[CLS]` al inicio (representacion del texto completo) y `[SEP]` al final.
- Padding: se rellena con `[PAD]` (valor 0) hasta llegar a la longitud maxima.

#### Attention Mask (Mascara de Atencion)
Un vector de 1s y 0s que indica cuales tokens son reales (1) y cuales son padding (0). BERT ignora las posiciones con 0.

#### Pooler Output
BERT produce un vector de 768 dimensiones que resume todo el texto. Es la representacion del token `[CLS]` pasada por una capa densa + activacion tanh. Este vector es lo que usamos para clasificar.

### Transfer Learning

Transfer learning significa tomar un modelo ya entrenado en una tarea grande (BERT aprendio a entender espanol) y adaptarlo a nuestra tarea especifica (clasificar tickets).

#### Feature Extraction (todo congelado)
- Se congelan TODOS los pesos de BERT (no se actualizan).
- BERT actua como un extractor de features: convierte texto en vectores ricos.
- Solo se entrena una capa clasificadora encima.
- Ventaja: rapido, menos memoria, menos riesgo de overfitting.
- Desventaja: BERT no se adapta a nuestro dominio especifico.

#### Fine-Tuning Parcial (lo que usamos)
- Se congelan las primeras 10 capas de BERT (features generales del idioma).
- Se descongelan las ultimas 2 capas (layers 10 y 11) + pooler.
- Las capas inferiores capturan features generales (sintaxis, estructura).
- Las capas superiores capturan features mas especificas que se adaptan a nuestro dominio.
- Ventaja: BERT adapta sus representaciones a tickets de CRM.
- Desventaja: mas lento, necesita LR bajo (2e-5) para no destruir los pesos pre-entrenados.

### Class Weights (Pesos de Clase)

Cuando el dataset esta desbalanceado (algunas clases tienen muchos mas ejemplos que otras), el modelo tiende a predecir las clases mayoritarias. Los class weights compensan esto:
- Clases con **pocos ejemplos** reciben un **peso alto** → errores en estas clases cuestan mas.
- Clases con **muchos ejemplos** reciben un **peso bajo** → errores cuestan menos.
- Formula: `peso = n_total / (n_clases × n_ejemplos_clase)`.

---

## 4. El Pipeline Completo

### Flujo de datos

```
CSV (texto + etiqueta)
    ↓
Limpieza (eliminar nulos, agrupar etiquetas)
    ↓
Filtrado (eliminar clases con pocos ejemplos)
    ↓
Label Encoding (texto → numeros)
    ↓
Train/Val/Test Split (70%/15%/15%)
    ↓
Tokenizacion con BETO (texto → tokens → IDs numericos)
    ↓
PyTorch Dataset + DataLoader (organiza en batches)
    ↓
Modelo (BERT parcialmente descongelado + clasificador)
    ↓
Entrenamiento (forward → loss → backward → update)
    ↓
Evaluacion (accuracy, precision, recall, F1, top-k)
    ↓
Prediccion (clasificar textos nuevos)
```

### Ciclo de entrenamiento

Cada epoca sigue este ciclo:

1. **Forward pass**: el texto tokenizado pasa por BERT y la capa clasificadora, produciendo logits (predicciones sin normalizar).
2. **Calcular loss**: CrossEntropyLoss compara los logits con la etiqueta real.
3. **Backward pass**: se calculan los gradientes de la loss respecto a cada parametro entrenable.
4. **Gradient clipping**: se limita la magnitud de los gradientes.
5. **Optimizer step**: AdamW actualiza los pesos usando los gradientes.
6. **Scheduler step**: se ajusta el learning rate.
7. **Validacion**: al final de cada epoca, se evalua en el set de validacion (sin actualizar pesos).
8. **Checkpoint**: se guarda el modelo si la val_loss mejoro.

---

## 5. Codigo Celda por Celda

### Celda 0: Introduccion (Markdown)

Descripcion general del proyecto. Clasifica tickets de CRM usando la columna `descripcion` para predecir `areas_resolutora`. Usa transfer learning con BETO.

---

### Celda 1: Instalar dependencias (Markdown)

Instruccion para instalar las librerias necesarias.

---

### Celda 2: Instalacion de paquetes

```python
# Instala todas las librerias necesarias desde requirements.txt
# !pip install -r requirements.txt -q
```

**Que hace**: Instala las dependencias del proyecto. Esta comentado porque ya estan instaladas.

---

### Celda 3: Imports y configuracion (Markdown)

Titulo de la seccion de imports.

---

### Celda 4: Imports - Librerias basicas

```python
import os       # Interactuar con el sistema de archivos (rutas, directorios)
import warnings # Silenciar advertencias que no son criticas

import pandas as pd  # Leer y manipular datos tabulares (CSV, DataFrames)
import numpy as np   # Operaciones numericas eficientes con arrays
```

**Que hace**: Importa las librerias fundamentales.
- `os`: para manejar rutas de archivos y directorios (ej: crear carpeta de checkpoints).
- `warnings`: para silenciar advertencias molestas que no afectan el funcionamiento.
- `pandas`: la libreria principal para manipular datos tabulares. Un DataFrame es como una hoja de Excel en Python.
- `numpy`: para operaciones matematicas eficientes con arrays y matrices.

---

### Celda 5: Imports - Machine Learning (sklearn)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
```

**Que hace**: Importa herramientas de scikit-learn.
- `train_test_split`: divide datos en conjuntos de train/validation/test manteniendo la proporcion de clases.
- `LabelEncoder`: convierte etiquetas de texto ("Call Center", "Sucursal") a numeros (0, 1, 2...) que el modelo puede procesar.
- Las metricas (`accuracy_score`, `precision_score`, etc.) se usan al final para evaluar que tan bien clasifica el modelo.

---

### Celda 6: Imports - Deep Learning y Transformers

```python
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')
```

**Que hace**: Importa las librerias de deep learning y utilidades.
- `torch` (PyTorch): el framework de deep learning. Maneja tensores, redes neuronales, y entrenamiento.
- `Dataset` y `DataLoader`: clases de PyTorch para organizar datos y entregarlos en batches al modelo.
- `BertTokenizer`: convierte texto a tokens que BERT entiende.
- `BertModel`: el modelo BERT pre-entrenado (sin capa de clasificacion, la agregamos nosotros).
- `get_linear_schedule_with_warmup`: scheduler que sube el LR gradualmente y despues lo baja.
- `matplotlib` y `seaborn`: para crear graficos (curvas de entrenamiento, matriz de confusion).
- `tqdm`: barra de progreso visual en el notebook.

---

### Celda 7: Verificar GPU (Markdown)

Explica que BERT es un modelo grande y que la GPU es necesaria para entrenamiento rapido.

---

### Celda 8: Verificar GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Dispositivo seleccionado: {device}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('ADVERTENCIA: No se detecto GPU. El entrenamiento sera mucho mas lento.')
```

**Que hace**: Detecta si hay una GPU NVIDIA disponible con soporte CUDA.
- `torch.cuda.is_available()`: devuelve True si hay GPU compatible.
- `device`: sera `'cuda'` (GPU) o `'cpu'`. Todos los tensores y el modelo se moveran a este dispositivo.
- Si hay GPU, muestra el nombre y memoria total.
- Sin GPU, el entrenamiento puede tardar 10-50x mas.

---

### Celda 9: Cargar datos (Markdown)

Titulo de la seccion de carga de datos.

---

### Celda 10: Cargar CSV

```python
df = pd.read_csv('test.csv', usecols=['descripcion', 'areas_resolutora'])

print(f'Filas totales en el dataset: {len(df)}')
df.head()
```

**Que hace**: Lee el archivo CSV con los datos de tickets.
- `usecols`: solo carga las 2 columnas que necesitamos para ahorrar memoria.
- `descripcion`: el texto del ticket (feature/entrada).
- `areas_resolutora`: a que area pertenece (label/etiqueta a predecir).
- `df.head()`: muestra las primeras 5 filas para inspeccionar los datos.

---

### Celda 11: Limpiar datos nulos

```python
print('Valores nulos por columna:')
print(df.isnull().sum())

df = df.dropna(subset=['descripcion', 'areas_resolutora'])
df = df.reset_index(drop=True)

print(f'\nFilas despues de limpiar nulos: {len(df)}')
print(f'Clases unicas (areas_resolutora): {df["areas_resolutora"].nunique()}')
```

**Que hace**: Elimina filas con valores faltantes.
- `isnull().sum()`: cuenta cuantos valores nulos hay en cada columna.
- `dropna(subset=...)`: elimina filas donde alguna de las columnas especificadas sea nula. No podemos entrenar sin texto o sin etiqueta.
- `reset_index(drop=True)`: reordena los indices de 0 a N despues de eliminar filas.
- `nunique()`: cuenta cuantas clases unicas hay.

---

### Celda 12: Limpiar etiquetas

```python
df['areas_resolutora'] = df['areas_resolutora'].str.split(', ').str[-1]
```

**Que hace**: Si la columna `areas_resolutora` tiene multiples valores separados por coma (ej: "Departamento, Subarea"), toma solo el ultimo elemento. Esto simplifica las etiquetas.

---

### Celda 13: Verificar valores unicos

```python
df.nunique()
```

**Que hace**: Muestra cuantos valores unicos tiene cada columna despues de la limpieza.

---

### Celda 14: Funcion para agrupar etiquetas

```python
def group_labels(label):
    if 'sucursal' in str(label).lower():
        return 'sucursales'
    elif 'cac' in str(label).lower():
        return 'CAC'
    else:
        return label
```

**Que hace**: Define una funcion que agrupa etiquetas similares.
- Todas las etiquetas que contengan "sucursal" (sin importar mayusculas/minusculas) se agrupan como "sucursales".
- Todas las que contengan "cac" se agrupan como "CAC".
- El resto se mantiene igual.
- `str(label).lower()`: convierte a minusculas para hacer la comparacion insensible a mayusculas.

---

### Celda 15: Aplicar agrupacion

```python
df['labels'] = df['areas_resolutora'].apply(group_labels)
```

**Que hace**: Aplica la funcion `group_labels` a cada fila del DataFrame.
- `.apply()`: ejecuta una funcion sobre cada valor de la columna.
- Crea una nueva columna `labels` con las etiquetas agrupadas.

---

### Celda 16: Inspeccionar DataFrame

```python
df
```

**Que hace**: Muestra el DataFrame completo para verificar que las transformaciones son correctas.

---

### Celda 17: Verificar valores unicos post-agrupacion

```python
df.nunique()
```

**Que hace**: Verifica cuantas clases unicas quedaron despues de agrupar.

---

### Celda 18: Exploracion de clases (primera version, MIN_SAMPLES=100)

```python
class_counts = df['labels'].value_counts()

print('classes:')
print(class_counts.head(60))

print(f'\nClases con menos de 10 ejemplos: {(class_counts < 10).sum()}')

MIN_SAMPLES = 100

valid_classes = class_counts[class_counts >= MIN_SAMPLES].index
df_filter = df[df['labels'].isin(valid_classes)]
df_filter = df_filter.reset_index(drop=True)

print(f'\nClases que se mantienen: {len(valid_classes)}')
print(f'Filas despues de filtrar clases minoritarias: {len(df)}')
```

**Que hace**: Primera exploracion del balance de clases con MIN_SAMPLES=100.
- `value_counts()`: cuenta cuantos ejemplos tiene cada clase, ordenado de mayor a menor.
- Muestra las 60 clases mas frecuentes.
- Filtra clases con menos de 100 ejemplos.
- **NOTA**: Esta celda tiene un bug — imprime `len(df)` en vez de `len(df_filter)` al final.

---

### Celda 19: Filtrado definitivo (MIN_SAMPLES=1000)

```python
class_counts = df['labels'].value_counts()

print('classes:')
print(class_counts.head(60))

print(f'\nClases con menos de 10 ejemplos: {(class_counts < 10).sum()}')

MIN_SAMPLES = 1000

valid_classes = class_counts[class_counts >= MIN_SAMPLES].index
df_filter = df[df['labels'].isin(valid_classes)]
df_filter = df_filter.reset_index(drop=True)

print(f'\nClases que se mantienen: {len(valid_classes)}')
print(f'Filas despues de filtrar clases minoritarias: {len(df_filter)}')
print(df_filter['labels'].unique())
```

**Que hace**: Filtrado final de clases minoritarias.
- `MIN_SAMPLES = 1000`: solo mantiene clases con al menos 1000 ejemplos. Esto es agresivo pero garantiza que cada clase tenga suficientes datos para aprender.
- `isin(valid_classes)`: filtra el DataFrame para quedarse solo con las clases que tienen suficientes ejemplos.
- Imprime las clases que sobrevivieron el filtro.

**Por que esto es importante**: El modelo no puede aprender patrones de una clase con muy pocos ejemplos. Clases con 5-10 ejemplos solo agregan ruido.

---

### Celda 20: Visualizar distribucion de clases

```python
plt.figure(figsize=(14, 6))

top_classes = df_filter['labels'].value_counts().head(15)
top_classes.plot(kind='barh', color='steelblue')

plt.title('Top 15 clases mas frecuentes (areas_resolutora)')
plt.xlabel('Cantidad de ejemplos')
plt.ylabel('Area resolutora')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Que hace**: Crea un grafico de barras horizontales con las 15 clases mas frecuentes.
- `kind='barh'`: barras horizontales para que los nombres largos se lean bien.
- `invert_yaxis()`: pone la clase mas frecuente arriba.
- Este grafico es clave para entender el desbalance de clases: si una clase tiene 10,000 ejemplos y otra 100, el modelo va a tener problemas.

---

### Celda 21: Codificar etiquetas (Markdown)

Titulo de la seccion de codificacion y division de datos.

---

### Celda 22: Label Encoding

```python
label_encoder = LabelEncoder()

df_filter['label_encoder'] = label_encoder.fit_transform(df_filter['labels'])

num_classes = len(label_encoder.classes_)

print(f'Numero total de clases: {num_classes}')
print('\nMapeo clase -> numero (primeras 10):')

for i, cls in enumerate(label_encoder.classes_[:60]):
    print(f'  {i}: {cls}')
```

**Que hace**: Convierte las etiquetas de texto a numeros.
- `LabelEncoder`: aprende todas las clases unicas y asigna un numero a cada una.
- `fit_transform()`: hace dos cosas en un paso:
  1. `fit`: aprende que clases existen (ej: "Call Center"=0, "Sucursales"=1).
  2. `transform`: convierte cada etiqueta a su numero.
- `num_classes`: el numero total de clases, necesario para la ultima capa del modelo.
- Los modelos de deep learning necesitan numeros, no texto.

---

### Celda 23: Train/Val/Test Split

```python
texts = df_filter['descripcion'].tolist()
labels = df_filter['label_encoder'].tolist()

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels,
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels,
)

print(f'Ejemplos de entrenamiento (train): {len(train_texts)}')
print(f'Ejemplos de validacion (val): {len(val_texts)}')
print(f'Ejemplos de prueba (test): {len(test_texts)}')
```

**Que hace**: Divide los datos en 3 conjuntos.
- Primera division: 70% train, 30% temporal.
- Segunda division: el 30% temporal se divide en 15% validation + 15% test.
- `stratify=labels`: mantiene la misma proporcion de cada clase en todos los conjuntos. Si "Call Center" es el 30% del total, sera el 30% en train, val y test.
- `random_state=42`: semilla para reproducibilidad (siempre la misma division).

---

### Celda 24: Tokenizacion con BETO (Markdown)

Explica que BETO es BERT para espanol y que el tokenizador convierte texto en tokens.

---

### Celda 25: Cargar tokenizador

```python
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

ejemplo = 'El cliente solicita el bloqueo de su tarjeta de credito'
tokens = tokenizer.tokenize(ejemplo)

print(f'Texto original: {ejemplo}')
print(f'Tokens: {tokens}')
print(f'Cantidad de tokens: {len(tokens)}')
```

**Que hace**: Carga el tokenizador de BETO y muestra un ejemplo.
- `MODEL_NAME`: identificador del modelo BETO en Hugging Face.
- `from_pretrained()`: descarga el tokenizador (vocabulario y reglas de tokenizacion).
- `tokenize()`: divide el texto en sub-palabras. Ejemplo: "autenticacion" → ["autent", "##icacion"]. Los tokens con `##` son continuaciones de la palabra anterior.

---

### Celda 26: Definir MAX_LENGTH

```python
MAX_LENGTH = 128

sample_lengths = []

for text in train_texts[:1000]:
    tokens = tokenizer.tokenize(str(text))
    sample_lengths.append(len(tokens))

print(f'Longitud promedio de tokens: {np.mean(sample_lengths):.0f}')
print(f'Longitud mediana de tokens: {np.median(sample_lengths):.0f}')
print(f'Longitud maxima de tokens: {np.max(sample_lengths)}')
print(f'Percentil 95: {np.percentile(sample_lengths, 95):.0f}')
print(f'Textos que se truncaran (>{MAX_LENGTH} tokens): {sum(1 for l in sample_lengths if l > MAX_LENGTH)} de 1000')
```

**Que hace**: Define la longitud maxima de tokens y analiza la distribucion.
- `MAX_LENGTH = 128`: BERT acepta hasta 512 tokens, pero usamos 128 porque la mediana de nuestros textos es ~82 tokens.
- Textos mas largos que 128 se **truncan** (se cortan).
- Textos mas cortos se **paddean** (se rellenan con ceros).
- Usar 128 en vez de 256 reduce el tiempo de procesamiento a la mitad y usa la mitad de memoria GPU.

---

### Celda 27: Crear Dataset de PyTorch (Markdown)

Explica que PyTorch necesita un objeto Dataset para organizar los datos.

---

### Celda 28: Clase TicketDataset

```python
class TicketDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        """Inicializa el dataset pre-tokenizando TODOS los textos de una vez."""
        self.labels = labels

        print(f'Pre-tokenizando {len(texts)} textos...')
        self.encodings = tokenizer(
            [str(t) for t in texts],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        print('Tokenizacion completa!')

    def __len__(self):
        """Devuelve cuantos ejemplos tiene el dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Devuelve un ejemplo ya tokenizado dado su indice."""
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

**Que hace**: Define como se organizan y entregan los datos al modelo.
- Hereda de `torch.utils.data.Dataset` — PyTorch requiere que implementemos `__len__` y `__getitem__`.
- **Pre-tokenizacion en `__init__`**: tokeniza TODOS los textos una sola vez al crear el dataset. Esto es mucho mas rapido que tokenizar en cada `__getitem__` (que se ejecutaria en cada batch, cada epoca).
- `add_special_tokens=True`: agrega `[CLS]` al inicio y `[SEP]` al final.
- `padding='max_length'`: rellena textos cortos con ceros hasta `max_length`.
- `truncation=True`: corta textos largos a `max_length`.
- `return_attention_mask=True`: genera la mascara que indica tokens reales (1) vs padding (0).
- `return_tensors='pt'`: devuelve tensores de PyTorch (no listas de Python).
- `__getitem__`: simplemente busca los tensores pre-calculados por indice — sin tokenizar nada.

---

### Celda 29: Crear Datasets y DataLoaders

```python
train_dataset = TicketDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = TicketDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
test_dataset = TicketDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

sample_batch = next(iter(train_loader))

print(f'Shape input_ids: {sample_batch["input_ids"].shape}')
print(f'Shape attention_mask: {sample_batch["attention_mask"].shape}')
print(f'Shape labels: {sample_batch["label"].shape}')
```

**Que hace**: Crea los 3 datasets y sus DataLoaders.
- `TicketDataset(...)`: crea un dataset para cada conjunto (train, val, test). La tokenizacion ocurre aqui.
- `BATCH_SIZE = 16`: procesa 16 textos a la vez. Si hay error de memoria GPU, reducir a 8 o 4.
- `DataLoader`: iterador que entrega datos en batches.
  - `shuffle=True` en train: mezcla los datos cada epoca para que el modelo no memorice el orden.
  - `shuffle=False` en val/test: no es necesario mezclar para evaluacion.
- `next(iter(train_loader))`: toma el primer batch para verificar que las dimensiones son correctas.
- Shapes esperadas: `[16, 128]` para input_ids y attention_mask, `[16]` para labels.

---

### Celda 30: Modelo (Markdown)

Explica el concepto de fine-tuning parcial: congelar capas 0-9 y descongelar capas 10-11 de BERT.

---

### Celda 31: Clase BERTClassifier

```python
class BERTClassifier(torch.nn.Module):

    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        super(BERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)

        # Congelar todo primero
        for param in self.bert.parameters():
            param.requires_grad = False

        # Descongelar ultimas 2 capas + pooler
        for name, param in self.bert.named_parameters():
            if 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'pooler' in name:
                param.requires_grad = True

        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = bert_output.pooler_output
        logits = self.classifier(pooled_output)

        return logits
```

**Que hace**: Define la arquitectura del modelo de clasificacion.
- Hereda de `torch.nn.Module` — la clase base para todos los modelos en PyTorch.
- **`__init__`** (constructor):
  1. Carga BETO pre-entrenado con `from_pretrained()`.
  2. Congela TODOS los parametros (`requires_grad = False`).
  3. Descongela selectivamente las capas 10, 11 y el pooler (`requires_grad = True`).
  4. Define la capa clasificadora: Dropout(0.1) → Linear(768, num_classes).
- **`forward`** (propagacion hacia adelante):
  1. Pasa el texto por BERT completo. Las capas 0-9 estan congeladas (no acumulan gradientes), las capas 10-11 si.
  2. Toma el `pooler_output` (vector de 768 dimensiones que resume el texto).
  3. Pasa el vector por la capa clasificadora para obtener los logits (un valor por cada clase).
- **Logits**: valores sin normalizar. La clase con el valor mas alto es la prediccion. CrossEntropyLoss aplica softmax internamente.

---

### Celda 32: Instanciar el modelo

```python
model = BERTClassifier(
    model_name=MODEL_NAME,
    num_classes=num_classes,
    dropout_rate=0.1,
)

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f'Parametros totales: {total_params:,}')
print(f'Parametros congelados (BERT capas 0-9): {frozen_params:,}')
print(f'Parametros entrenables (capas 10-11 + pooler + clasificador): {trainable_params:,}')
print(f'Porcentaje que se entrena: {trainable_params/total_params*100:.2f}%')
```

**Que hace**: Crea una instancia del modelo y la mueve a GPU.
- `BERTClassifier(...)`: crea el modelo con la configuracion especificada.
- `.to(device)`: mueve todos los tensores del modelo a la GPU (o CPU).
- Cuenta y muestra los parametros:
  - **Totales**: ~110 millones (todos los de BERT + clasificador).
  - **Congelados**: ~95 millones (capas 0-9 de BERT).
  - **Entrenables**: ~15 millones (capas 10-11, pooler, clasificador).

---

### Celda 33: Configurar entrenamiento (Markdown)

Titulo de la seccion de configuracion.

---

### Celda 34: Hiperparametros

```python
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

**Que hace**: Define los hiperparametros del entrenamiento.
- `NUM_EPOCHS = 5`: 5 pasadas completas por los datos. Con fine-tuning, 3-5 epocas suelen ser suficientes.
- `LEARNING_RATE = 2e-5`: tasa de aprendizaje baja, necesaria para no destruir los pesos pre-entrenados de BERT. Este es el rango recomendado (2e-5 a 5e-5) para fine-tuning de transformers.
- `CHECKPOINT_DIR`: carpeta donde se guardan los modelos durante el entrenamiento.

---

### Celda 35: Funcion de perdida, optimizador y scheduler

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f'Class weights (min: {class_weights.min():.2f}, max: {class_weights.max():.2f})')

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

total_steps = len(train_loader) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)

print(f'Total de pasos de entrenamiento: {total_steps:,}')
print(f'Pasos de warmup: {int(total_steps * 0.1):,}')
```

**Que hace**: Configura los 3 componentes clave del entrenamiento.

1. **Class Weights (Pesos de clase)**:
   - `compute_class_weight('balanced', ...)`: calcula pesos inversamente proporcionales a la frecuencia de cada clase.
   - Clases con pocos ejemplos reciben peso alto → el modelo les presta mas atencion.
   - Se pasan a `CrossEntropyLoss(weight=...)` para que la funcion de perdida sea balanceada.

2. **Criterion (Funcion de perdida)**:
   - `CrossEntropyLoss`: combina LogSoftmax + NLLLoss. Mide que tan lejos estan las predicciones de las etiquetas reales.
   - Con `weight=class_weights`: penaliza mas los errores en clases minoritarias.

3. **Optimizer (Optimizador)**:
   - `AdamW`: variante de Adam con weight decay (regularizacion L2 corregida).
   - `filter(lambda p: p.requires_grad, ...)`: solo optimiza parametros no congelados.
   - `weight_decay=0.01`: agrega regularizacion para prevenir overfitting.

4. **Scheduler**:
   - `total_steps`: numero total de batches en todo el entrenamiento.
   - `warmup`: primeros 10% de pasos, el LR sube de 0 a 2e-5 gradualmente.
   - `decay`: el 90% restante, el LR baja linealmente de 2e-5 a 0.

---

### Celda 36: Checkpoint (Markdown)

Explica como detener y reanudar el entrenamiento usando checkpoints.

---

### Celda 37: Funciones de checkpoint

```python
def save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, val_acc, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer=None, scheduler=None, path=None):
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')

    if not os.path.exists(path):
        print(f'No se encontro checkpoint en: {path}')
        return 0, 0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f'Checkpoint cargado: epoca {checkpoint["epoch"]}, paso {checkpoint["step"]}')
    print(f'Val Loss: {checkpoint["val_loss"]:.4f}, Val Acc: {checkpoint["val_acc"]:.4f}')

    return checkpoint['epoch'], checkpoint['step']
```

**Que hace**: Define funciones para guardar y cargar el estado del entrenamiento.

- **`save_checkpoint`**: guarda TODO lo necesario para reanudar:
  - `model.state_dict()`: todos los pesos del modelo.
  - `optimizer.state_dict()`: estado del optimizador (momentums, learning rates adaptativos).
  - `scheduler.state_dict()`: en que paso del schedule estamos.
  - `epoch`, `step`: para saber desde donde continuar.
  - `val_loss`, `val_acc`: para saber que tan bueno era el modelo.

- **`load_checkpoint`**: restaura el estado completo:
  - Carga los pesos en el modelo.
  - Si se pasan optimizer/scheduler, restaura sus estados tambien.
  - `map_location=device`: asegura que se cargue en el dispositivo correcto (si guardaste en GPU y cargas en CPU, o viceversa).

---

### Celda 38: Entrenamiento (Markdown)

Titulo de la seccion. Explica que se puede reanudar ejecutando la celda de nuevo.

---

### Celda 39: Preparar entrenamiento

```python
resume_path = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt')
start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, resume_path)

train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')

print(f'\nIniciando entrenamiento desde epoca {start_epoch}...')
print(f'Epocas: {start_epoch} -> {NUM_EPOCHS}')
print(f'Batches por epoca: {len(train_loader)}')
print(f'Dispositivo: {device}')
```

**Que hace**: Prepara el entrenamiento, intentando reanudar si hay un checkpoint.
- Intenta cargar `last_checkpoint.pt`. Si no existe, empieza desde epoca 0.
- Inicializa listas vacias para el historial de metricas.
- `best_val_loss = float('inf')`: cualquier loss sera mejor que infinito, asi que el primer modelo siempre se guarda.

---

### Celda 40: Loop de entrenamiento

```python
for epoch in range(start_epoch, NUM_EPOCHS):

    # --- FASE DE ENTRENAMIENTO ---
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0

    train_bar = tqdm(train_loader, desc=f'Epoca {epoch+1}/{NUM_EPOCHS} [Train]')

    for batch_idx, batch in enumerate(train_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()                          # 1. Limpiar gradientes
        logits = model(input_ids, attention_mask)      # 2. Forward pass
        loss = criterion(logits, labels)               # 3. Calcular loss
        loss.backward()                                # 4. Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 5. Gradient clipping
        optimizer.step()                               # 6. Actualizar pesos
        scheduler.step()                               # 7. Actualizar LR

        total_train_loss += loss.item()
        _, predicted = torch.max(logits, dim=1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        train_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{train_correct/train_total:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
        })

        global_step += 1
        if global_step % 500 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, 0, 0,
                          os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # --- FASE DE VALIDACION ---
    model.eval()
    total_val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoca {epoch+1}/{NUM_EPOCHS} [Val]'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = val_correct / val_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f'\nEpoca {epoch+1}/{NUM_EPOCHS}')
    print(f'  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}')

    save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                   avg_val_loss, val_acc,
                   os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                       avg_val_loss, val_acc,
                       os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        print(f'  >>> Nuevo mejor modelo guardado! (Val Loss: {avg_val_loss:.4f})')

print(f'\nMejor Val Loss: {best_val_loss:.4f}')
```

**Que hace**: El loop principal de entrenamiento. Esta es la celda mas importante.

**Fase de entrenamiento (por cada epoca):**
1. `model.train()`: activa dropout y batch normalization en modo entrenamiento.
2. Para cada batch:
   - Mueve datos a GPU con `.to(device)`.
   - `optimizer.zero_grad()`: limpia gradientes del paso anterior (si no, se acumulan).
   - `model(input_ids, attention_mask)`: forward pass — produce logits.
   - `criterion(logits, labels)`: calcula la loss (que tan mal predijo).
   - `loss.backward()`: backward pass — calcula gradientes.
   - `clip_grad_norm_(..., max_norm=1.0)`: limita gradientes para estabilidad.
   - `optimizer.step()`: actualiza pesos del modelo.
   - `scheduler.step()`: ajusta el learning rate.
3. Guarda checkpoint cada 500 pasos (por seguridad).

**Fase de validacion (al final de cada epoca):**
1. `model.eval()`: desactiva dropout (usa todas las neuronas).
2. `torch.no_grad()`: no calcula gradientes (ahorra memoria, mas rapido).
3. Evalua en todos los batches de validacion.
4. Calcula loss y accuracy promedio.

**Guardado de modelos:**
- `last_checkpoint.pt`: siempre se guarda (para reanudar).
- `best_model.pt`: solo se guarda si la val_loss mejoro (el mejor modelo encontrado).

---

### Celda 41: Graficas de entrenamiento

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(val_losses, label='Val Loss', marker='s')
axes[0].set_title('Loss por Epoca')
axes[0].set_xlabel('Epoca')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accuracies, label='Val Accuracy', marker='s', color='green')
axes[1].set_title('Accuracy de Validacion por Epoca')
axes[1].set_xlabel('Epoca')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Que hace**: Grafica las curvas de entrenamiento en 2 subplots.

**Subplot 1 — Loss:**
- Si train_loss y val_loss bajan juntas: el modelo esta generalizando bien.
- Si train_loss baja pero val_loss sube: **overfitting** — el modelo memoriza.
- Si ambas son altas: **underfitting** — necesita mas capacidad o datos.

**Subplot 2 — Accuracy de validacion:**
- Deberia subir con cada epoca.
- Si se estanca: el modelo ya aprendio lo que puede con esta configuracion.

---

### Celda 42: Evaluacion (Markdown)

Explica que se carga el mejor modelo y se evalua en datos que nunca vio.

---

### Celda 43: Evaluacion en test set

```python
load_checkpoint(model, path=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
model.eval()

all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluando en test set'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits, dim=1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_probabilities = np.array(all_probabilities)
```

**Que hace**: Evalua el mejor modelo en el conjunto de test.
- Carga `best_model.pt` (el que tuvo menor val_loss durante el entrenamiento).
- `model.eval()`: desactiva dropout.
- `torch.no_grad()`: no calcula gradientes (solo queremos predicciones).
- Para cada batch:
  - Forward pass para obtener logits.
  - `softmax`: convierte logits a probabilidades (suman 1).
  - `torch.max`: obtiene la clase con mayor probabilidad.
- `.cpu().numpy()`: mueve resultados de GPU a CPU y convierte a arrays de numpy.

---

### Celda 44: Metricas (Markdown)

Tabla explicando cada metrica: accuracy, precision, recall, F1-score, top-k accuracy.

---

### Celda 45: Calcular metricas

```python
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
top3_acc = top_k_accuracy_score(all_labels, all_probabilities, k=3)
top5_acc = top_k_accuracy_score(all_labels, all_probabilities, k=5)

print(f'Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'Precision (macro):    {precision:.4f} ({precision*100:.2f}%)')
print(f'Recall (macro):       {recall:.4f} ({recall*100:.2f}%)')
print(f'F1-Score (macro):     {f1_macro:.4f} ({f1_macro*100:.2f}%)')
print(f'F1-Score (weighted):  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)')
print(f'Top-3 Accuracy:       {top3_acc:.4f} ({top3_acc*100:.2f}%)')
print(f'Top-5 Accuracy:       {top5_acc:.4f} ({top5_acc*100:.2f}%)')
```

**Que hace**: Calcula todas las metricas de evaluacion.
- `accuracy_score`: porcentaje de predicciones correctas.
- `precision_score(average='macro')`: precision promedio de todas las clases (sin ponderar).
- `recall_score(average='macro')`: recall promedio de todas las clases.
- `f1_score(average='macro')`: F1 promedio — mejor metrica para datos desbalanceados.
- `f1_score(average='weighted')`: F1 ponderado por frecuencia de clase.
- `top_k_accuracy_score(k=3)`: si la respuesta correcta esta en las top 3 predicciones.
- `top_k_accuracy_score(k=5)`: si esta en las top 5.
- `zero_division=0`: si una clase no tiene predicciones, devuelve 0 en vez de error.

---

### Celda 46: Reporte por clase

```python
unique_labels = np.unique(all_labels)
class_names = label_encoder.inverse_transform(unique_labels)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0,
)

print(report)
```

**Que hace**: Muestra precision, recall y F1 para CADA clase individualmente.
- `inverse_transform`: convierte los numeros de vuelta a nombres de clase.
- `classification_report`: genera una tabla con metricas por clase.
- **support**: cantidad de ejemplos de esa clase en test.
- Si precision alta pero recall bajo: el modelo es conservador con esa clase (predice poco pero acertado).
- Si recall alto pero precision baja: el modelo predice esa clase demasiado (muchos falsos positivos).

---

### Celda 47: Matriz de confusion

```python
top_n = 40
top_classes_idx = pd.Series(all_labels).value_counts().head(top_n).index.tolist()
mask = np.isin(all_labels, top_classes_idx)
filtered_labels = all_labels[mask]
filtered_preds = all_predictions[mask]
cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes_idx)
top_class_names = label_encoder.inverse_transform(top_classes_idx)

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=top_class_names, yticklabels=top_class_names)
plt.title(f'Matriz de Confusion (Top {top_n} clases)')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

**Que hace**: Visualiza la matriz de confusion como un heatmap.
- Solo muestra las top 40 clases (para que sea legible).
- **Diagonal**: predicciones correctas (numeros altos = bien).
- **Fuera de la diagonal**: errores. La celda [i, j] indica cuantos ejemplos de la clase i fueron predichos como clase j.
- Ejemplo: si la celda [Call Center, Sucursal] tiene 50, significa que 50 tickets de Call Center fueron mal clasificados como Sucursal.
- `cmap='Blues'`: colores azules — mas oscuro = mas ejemplos.

---

### Celda 48: Prediccion (Markdown)

Explica que la funcion permite clasificar textos nuevos sin reentrenar.

---

### Celda 49: Funcion de prediccion

```python
def predict(text, model, tokenizer, label_encoder, device, top_k=3):
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probabilities = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    top_classes = label_encoder.inverse_transform(top_indices)

    print(f'Texto: "{text[:100]}..."\n')
    print(f'Top {top_k} predicciones:')
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
        bar = '█' * int(prob * 30)
        print(f'  {i+1}. {cls}: {prob:.4f} ({prob*100:.1f}%) {bar}')

    return top_classes[0], dict(zip(top_classes, top_probs))
```

**Que hace**: Clasifica un texto nuevo y muestra las top-K predicciones con sus probabilidades.
1. Tokeniza el texto con el mismo tokenizador de BETO.
2. Mueve los tensores a GPU.
3. Forward pass sin gradientes.
4. Softmax para convertir logits en probabilidades.
5. `torch.topk`: obtiene las K clases con mayor probabilidad.
6. `inverse_transform`: convierte numeros de vuelta a nombres de clase.
7. Muestra una barra visual proporcional a la probabilidad.
8. Devuelve la clase predicha y un diccionario con probabilidades.

---

### Celda 50: Ejemplos de prediccion

```python
texto_1 = 'El cliente llama para solicitar el bloqueo de su tarjeta de credito porque fue robada'
clase_1, probs_1 = predict(texto_1, model, tokenizer, label_encoder, device)

texto_2 = 'No puedo ingresar a la aplicacion movil, me sale error de autenticacion'
clase_2, probs_2 = predict(texto_2, model, tokenizer, label_encoder, device)

texto_3 = 'Quiero saber el saldo de mi cuenta de ahorro y los ultimos movimientos'
clase_3, probs_3 = predict(texto_3, model, tokenizer, label_encoder, device)
```

**Que hace**: Prueba la funcion de prediccion con 3 textos de ejemplo.
- Texto 1: consulta sobre tarjeta de credito (deberia clasificar como el area de tarjetas).
- Texto 2: problema con app movil (deberia clasificar como area tecnologica).
- Texto 3: consulta sobre cuenta (deberia clasificar como area de cuentas).

---

### Celda 51: Reanudar entrenamiento (Markdown)

Instrucciones para detener, usar y reanudar el entrenamiento.

---

### Celda 52: Cargar modelo guardado

```python
model = BERTClassifier(model_name=MODEL_NAME, num_classes=num_classes)
model = model.to(device)
load_checkpoint(model, path=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
print('\nModelo listo para usar!')
```

**Que hace**: Carga el mejor modelo guardado sin necesidad de reentrenar.
1. Crea una nueva instancia del modelo con la misma arquitectura.
2. La mueve a GPU.
3. Carga los pesos del mejor checkpoint.
4. Ahora se puede usar la funcion `predict()` directamente.

---

### Celda 53: Reanudar entrenamiento (Markdown)

Instrucciones para entrenar mas epocas despues de probar el modelo.

---

### Celda 54: Reanudar entrenamiento

```python
model = BERTClassifier(model_name=MODEL_NAME, num_classes=num_classes)
model = model.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

EXTRA_EPOCHS = 5
extra_total_steps = len(train_loader) * EXTRA_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(extra_total_steps * 0.1),
    num_training_steps=extra_total_steps,
)

resume_epoch, resume_step = load_checkpoint(
    model, optimizer, scheduler,
    path=os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'),
)

print(f'\nReanudando desde epoca {resume_epoch}')
print(f'Entrenando {EXTRA_EPOCHS} epocas adicionales...')
```

**Que hace**: Prepara todo para continuar entrenando mas epocas.
1. Crea modelo, optimizador y scheduler nuevos.
2. Carga el ultimo checkpoint (restaura pesos, estado del optimizador y scheduler).
3. Define cuantas epocas adicionales entrenar.
4. Despues de ejecutar esta celda, hay que volver a la celda de entrenamiento (Seccion 10) y ejecutarla.

---

### Celda 55: Resumen (Markdown)

Resumen del pipeline completo y sugerencias para mejorar:
- Descongelar mas capas de BERT.
- Aumentar MAX_LENGTH si muchos textos se truncan.
- Limpiar datos (eliminar firmas, saludos).
- Class weighting (ya implementado).
- Data augmentation (parafrasear textos).
