
### Index en Elasticsearch

Un Index es una colección de documentos que comparten características similares.  
Conceptualmente, puede compararse con un corpus, ya que agrupa información relacionada.

- Un index está compuesto por documentos en formato JSON.
- Cada documento representa una unidad de información que Elasticsearch puede indexar, buscar y analizar.

### Conexión a Elasticsearch desde el ambiente  desa

Para crear y gestionar un index, primero debemos conectarnos a Elasticsearch.

Como en el ambiente de desarrollo Elasticsearch no está expuesto directamente, es necesario crear un **túnel SSH** para habilitar el acceso al puerto `9200` desde nuestra computadora local.

### Crear el túnel SSH

```bash
ssh -L 9200:localhost:9200 sysadmin@10.6.2.236
```

Esto redirige el puerto `9200` del servidor remoto al puerto `9200` de tu máquina local.

---

### Conexión desde Python

Una vez habilitado el túnel, podemos conectarnos usando la librería oficial de Elasticsearch para Python.

```python
from pprint import pprint
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
client_info = es.info()

print("Connected to Elasticsearch!")
pprint(client_info.body)
```

Si la conexión es exitosa, se mostrará la información del cluster.



### Creación de un Index

### Opción 1: Creación simple

Podemos crear un index sin especificar configuraciones adicionales.  
Antes de crearlo, es recomendable eliminarlo si ya existe para evitar errores.

```python
es.indices.delete(index="my_index", ignore_unavailable=True)
es.indices.create(index="my_index")
```

### Shards y Replicas

### Shards

Los **shards** son las particiones en las que se divide un index.  
Permiten distribuir los datos y las búsquedas entre distintos nodos, mejorando el rendimiento y la escalabilidad.

###  Replicas

Las **replicas** son copias de los shards primarios.  
Sirven para:

- Aumentar la disponibilidad
- Mejorar el rendimiento de lectura
- Proteger los datos ante fallos de nodos


### Opción 2: Creación con configuración personalizada

Podemos definir el número de shards y replicas en la sección `settings` al crear el index.

```python
es.indices.delete(index="my_index", ignore_unavailable=True)

es.indices.create(
    index="my_index",
    settings={
        "index": {
            "number_of_shards": 3,    # cantidad de shards (particiones)
            "number_of_replicas": 2   # cantidad de copias de los datos
        }
    }
)
```


### Insertar  documentos
Para insertar documentos en Elasticsearch, estos deben estar en formato JSON.  
Cada documento representa una fila de información dentro de un index.

- Todos los documentos de un mismo **index** deben compartir la misma estructura de campos 
- Elasticsearch utiliza un concepto llamado mapping, que define el **tipo de dato** de cada campo (texto, fecha, número, etc.).
- El mapping puede:
    - Ser **automático** (por defecto, Elasticsearch infiere los tipos).
    - Ser **manual**, definido explícitamente al crear el index.
```python
document = {
    'title': 'title',
    'text': 'text',
    'created_on': '2024-09-22',
}
response = es.index(index='my_index', body=document)
response
```
El cual te devuelve 
```json
ObjectApiResponse({'_index': 'my_index', 
'_id': 'OTO1oZsBeHoYWK3KSaDa', 
'_version': 1, 
'result': 'created', 
'_shards': {'total': 3, 'successful': 1, 'failed': 0},
 '_seq_no': 0, 
 '_primary_term': 1})
```

### Inserción de múltiples documentos

Podemos insertar múltiples documentos definiéndolos como una lista de objetos JSON, donde cada elemento de la lista representa un documento.

```python
data = [
    {
        "title": "Sample Title 1",
        "text": "This is the first sample document text.",
        "created_on": "2024-09-22",
    },
    {
        "title": "Sample Title 2",
        "text": "Here is another example of a document.",
        "created_on": "2024-09-22",
    },
    {
        "title": "Sample Title 3",
        "text": "The content of the third document goes here.",
        "created_on": "2024-09-22",
    },
]
```

Cada elemento de la lista es un documento independiente que será indexado.



### Función para insertar múltiples documentos

Podemos crear una función que inserte un documento y luego iterar sobre la lista.

```python
def insert_document(document):
    response = es.index(index="my_index", body=document)
    print(
        f"Document ID: {response['_id']} "
        f"is '{response['result']}' "
        f"and is split into {response['_shards']['total']} shards."
    )
    return response


for document in data:
    insert_document(document)
```



### Verificar el mapping del index

Una vez insertados los documentos, podemos consultar el mapping generado por Elasticsearch para ver los tipos de datos asignados a cada campo.

```python
from pprint import pprint

index_mapping = es.indices.get_mapping(index="my_index")
pprint(index_mapping["my_index"]["mappings"]["properties"])
```

### Resultado del mapping

```json
{
  "created_on": {
    "type": "date"
  },
  "text": {
    "type": "text",
    "fields": {
      "keyword": {
        "type": "keyword",
        "ignore_above": 256
      }
    }
  },
  "title": {
    "type": "text",
    "fields": {
      "keyword": {
        "type": "keyword",
        "ignore_above": 256
      }
    }
  }
}
```

### mapping manual

- El mapping puede definirse de **dos formas**:
    1. **Durante la creación del index**, usando el parámetro `mappings` en `indices.create`.
    2. **Después de crear el index**, utilizando `indices.put_mapping`.
        
- El mapping debe definirse antes de insertar documentos. Solo es posible agregar un mapping manual después de crear el index si el index no contiene ningún documento.
    


```python
from pprint import pprint

es.indices.delete(index="my_index", ignore_unavailable=True)
es.indices.create(index="my_index")
```

### Definición del mapping

```python
mapping = {
    "properties": {
        "created_on": {
            "type": "date"
        },
        "text": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 256
                }
            }
        },
        "title": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 256
                }
            }
        }
    }
}
```

### Aplicar el mapping al index

```python
es.indices.put_mapping(index="my_index", body=mapping)
```

---

### Verificar el mapping del index

```python
index_mapping = es.indices.get_mapping(index="my_index")
pprint(index_mapping["my_index"]["mappings"]["properties"])
```




### Eliminar documentos

Para eliminar un documento de Elasticsearch se utiliza el método `delete`, que requiere **dos parámetros obligatorios**:
- `index`: nombre del index donde se encuentra el documento
- `id`: identificador único del documento a eliminar
### Ejemplo

```python
try:
    response = es.delete(index="my_index", id="id_del_documento")
    print("Documento eliminado correctamente")
except Exception as e:
    print(f"Error al eliminar el documento: {e}")
```



### Obtener documentos

Para obtener un documento específico se utiliza el método `get`, que también necesita:

- `index`: nombre del index
- `id`: identificador único del documento
    

### Ejemplo

```python
try:
    response = es.get(index="my_index", id="id_del_documento")
    print(response["_source"])  # imprime el contenido del documento
except Exception as e:
    print(f"Error al obtener el documento: {e}")
```



### Contar documentos

El método `count` permite conocer la cantidad de documentos de un index.  
Opcionalmente, se puede usar un **query** para contar solo los documentos que cumplan ciertos criterios.

### Contar todos los documentos de un index

```python
response = es.count(index="my_index")
count = response["count"]

print(f"La cantidad de documentos en el index es {count}")
```

### Contar documentos según un criterio (ejemplo por fecha)

Si queremos contar documentos cuya fecha esté dentro de un rango específico, usamos un query `range`:

```python
query = {
    "range": {
        "created_on": {
            "gte": "2024-09-24",  # fecha mínima
            "lte": "2024-09-24",  # fecha máxima
            "format": "yyyy-MM-dd"
        }
    }
}

response = es.count(index="my_index", query=query)
count = response["count"]

print(f"La cantidad de documentos en el index según el criterio es {count}")
```

### Exists API
checkear si el index existe o no 
```python
response = es.indices.exists(index='my_index')
response.body
```
checkear si un documento exite en un index
```python
response = es.exists(index='my_index', id=document_ids[0])
response.body
```

### Update documents
EN script pones el campo que queres modificar
```python
from pprint import pprint

response = es.update(
    index="my_index",
    id=document_ids[0],
    script={
        "source": "ctx._source.title = params.title",
        "params": {
            "title": "New Title"
        }
    },
)
pprint(response.body)
```

### Upsert

```python
response = es.update(
    index="my_index",
    id="1",
    doc={
        "book_id": 1234,
        "book_name": "A book",
    },
    doc_as_upsert=True,
)
```


### Delete a field 
```python
response = es.update(
    index="my_index",
    id=document_ids[0],
    script={
        "source": "ctx._source.remove('new_field')",
    },
)
pprint(response.body)
```

###  Bulk API 

La Bulk API permite ejecutar múltiples operaciones (index, create, update, delete) en una sola llamada a Elasticsearch.  

### Ventajas
- Reduce la **latencia** al evitar múltiples llamadas HTTP.
- Permite realizar operaciones mixtas (insertar, actualizar y eliminar) en un solo request.
- Mejora el rendimiento en procesamiento masivo de datos.
    


### Ejemplo de uso de Bulk API

1. Insertar 3 documentos (`index`)
2. Actualizar 2 documentos (`update`)
3. Eliminar 1 documento (`delete`)
    

```python
from pprint import pprint

response = es.bulk(
    operations=[
        # ---- Insertar documentos ----
        {"index": {"_index": "my_index", "_id": "1"}},
        {"title": "Sample Title 1", "text": "This is the first sample document text.", "created_on": "2024-09-22"},

        {"index": {"_index": "my_index", "_id": "2"}},
        {"title": "Sample Title 2", "text": "Here is another example of a document.", "created_on": "2024-09-24"},

        {"index": {"_index": "my_index", "_id": "3"}},
        {"title": "Sample Title 3", "text": "The content of the third document goes here.", "created_on": "2024-09-24"},

        # ---- Actualizar documentos ----
        {"update": {"_index": "my_index", "_id": "1"}},
        {"doc": {"title": "New Title"}},

        {"update": {"_index": "my_index", "_id": "2"}},
        {"doc": {"new_field": "dummy_value"}},

        # ---- Eliminar documentos ----
        {"delete": {"_index": "my_index", "_id": "3"}},
    ],
)

# Ver la respuesta completa
pprint(response.body)
```



###  Interpretación de la respuesta

- La respuesta incluye cada operación individual y su resultado (`created`, `updated`, `deleted`, etc.).
- Si `errors` es `False`, significa que todas las operaciones se realizaron correctamente.
- Si `errors` es `True`, alguna de las operaciones falló y es posible revisar los detalles en la lista `items`.
    



Acá tenés la documentación **mejorada, clara y con mejor estructura**, manteniendo un lenguaje sencillo pero correcto para uso técnico 👌

---

### Search API en Elasticsearch

La Search API permite realizar búsquedas sobre uno o varios índices utilizando distintos tipos de consultas, desde búsquedas simples hasta consultas complejas con el Query DSL.

#### Parámetros 

- **index**
    - Podés buscar en:
        - Un solo index (`index_1`)
        - Varios indices separados por coma (`index_1,index_2`)
        - Usando comodines (`index*`)
        - Todos los indices (`_all`)
            
- **q**
    - Se utiliza para **búsquedas simples** en formato texto.
    - Es rápido, pero limitado.
        
- **query**
    - Se utiliza para **búsquedas complejas**.
    - Usa el **Query DSL de Elasticsearch** (recomendado para la mayoría de los casos).
        
- **timeout**
    - Tiempo máximo de espera para la búsqueda.
        
- **size**
    - Cantidad de documentos que se desean retornar.
        
- **from**
    - Punto de inicio para paginación (offset).
        

---

### Cómo realizar búsquedas?

### Buscar en un solo index

```python
response = es.search(
    index="index_1",
    body={
        "query": {
            "match_all": {}
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(f"Found {n_hits} documents in index_1")
```

---

### Buscar en múltiples indices

```python
response = es.search(
    index="index_1,index_2",
    body={
        "query": {
            "match_all": {}
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(f"Found {n_hits} documents in index_1 and index_2")
```

---

### Buscar usando comodines (wildcards)

Podés usar `*` para buscar en todos los indices que coincidan con un patrón.

```python
response = es.search(
    index="index*",
    body={
        "query": {
            "match_all": {}
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(
    "Found "
    f"{n_hits} documents in all indexes with name starting with 'index'"
)
```

---

### Buscar en todos los indices

```python
response = es.search(
    index="_all",
    body={
        "query": {
            "match_all": {}
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(f"Found {n_hits} documents in all indexes")
```


`n_hits` representa la cantidad total de documentos que coinciden con la búsqueda realizada, independientemente de cuántos documentos se devuelvan en la respuesta (limitados por `size`).
- En el cuerpo (`body`) de la búsqueda podés usar:
    - `q` → para búsquedas simples
    - `query` → para búsquedas más complejas y estructuradas
- Siempre que la búsqueda tenga cierta lógica (filtros, rangos, múltiples condiciones), se recomienda usar Query DSL
- Para grandes volúmenes de datos, usá paginación (`from`, `size`) o técnicas como search_after




### Query DSL (Domain Specific Language)

El Query DSL es el lenguaje que utiliza Elasticsearch para construir consultas complejas y estructuradas.  
Permite combinar distintos tipos de queries según el tipo de dato y el objetivo de la búsqueda.


### Tipos de queries más comunes

### `match`

- Se utiliza para **búsquedas de texto completo (full-text search)**.
- Elasticsearch analiza el texto (tokeniza, normaliza, etc.).
- Retorna documentos que **coinciden de forma aproximada** con el valor buscado.
- El campo debe estar mapeado como **`text`**.
    

### `term`

- Retorna documentos que contienen **exactamente** el valor indicado.
- **No analiza** el texto.
- El campo debe estar mapeado como:
    - `keyword`
    - `numeric`
    - `date`
    - `boolean`
### `range`

- Retorna documentos cuyos valores se encuentren **dentro de un rango**.
- Funciona con campos de tipo:
    - `date`
    - `numeric`
 Operadores comunes:

- `gte` → mayor o igual
- `lte` → menor o igual
- `gt` → mayor
- `lt` → menor
    


### Match query (búsqueda de texto)

```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "match": {
                "text": "document"
            }
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(f"Found {n_hits} documents in my_index")
```

---

### Term query (búsqueda exacta)

```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "term": {
                "created_on": "2024-09-22"
            }
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(f"Found {n_hits} documents in my_index")
```


Para texto, normalmente se usa `field.keyword`, no el campo `text`.

###  Range query (búsqueda por rango)

```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "range": {
                "created_on": {
                    "lte": "2024-09-23"
                }
            }
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(f"Found {n_hits} documents in my_index")
```

---

### Cláusulas compuestas (`bool`)

El query **`bool`** permite combinar múltiples queries utilizando **lógica booleana**.

### Operadores disponibles:

- **must** → todas las condiciones deben cumplirse
- **filter** → filtra resultados (no afecta scoring)
- **should** → al menos una condición debería cumplirse
- **must_not** → excluye documentos
    

`bool` puede combinar cualquier tipo de query (`match`, `term`, `range`, etc.).


### Ejemplo de query compuesto

En este ejemplo:
- Se busca texto que contenga `"third"`
- Y documentos creados en una fecha específica
    
```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": "third"
                        }
                    },
                    {
                        "range": {
                            "created_on": {
                                "gte": "2024-09-24",
                                "lte": "2024-09-24"
                            }
                        }
                    }
                ]
            }
        }
    }
)

n_hits = response["hits"]["total"]["value"]
print(f"Found {n_hits} documents in my_index")
```
Parametros
Time out /size/from
Time out es el maximo tiempo de duracion para que un query se ejecute si toma mas de x tiempo elastic search abprt the search
size es cuantos resultados va a retornar 
from se usa para paginacion cuantos documentos skipperar antes de buscar al abusqueda
agreegation 
seria hacer calculos con los datos ya sea average, max, min, count

### size and from
```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "match_all": {}
        },
        "size": 10,
        "from": 10
    },
)

for hit in response['hits']['hits']:
    print(hit['_source'])
```

### time out 
normalmente no se usa pero si tenes queries muy complejas podrias prever esto
```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "match": {
                "message": "search keyword"
            }
        },
        "timeout": "10s"
    },
)

response.body
```
### Agregation
```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "match_all": {}
        },
        "aggs": {
            "avg_age": {
                "avg": {
                    "field": "age"
                }
            }
        }
    }
)

average_age = response['aggregations']['avg_age']['value']
print(f"Average Age: {average_age}")
```

### Parámetros comunes en Search API

Elasticsearch provee varios parámetros que permiten controlar el comportamiento de las búsquedas, como el tiempo máximo de ejecución, la cantidad de resultados devueltos y la paginación.


### `timeout`

- Define el tiempo máximo que Elasticsearch puede tardar en ejecutar una búsqueda.
- Si la consulta supera ese tiempo, Elasticsearch interrumpe la búsqueda y retorna los resultados parciales disponibles. 
- Se expresa como una cadena de tiempo: `"5s"`, `"100ms"`, etc.
    

Normalmente no se utiliza, pero es útil para:

- Queries muy complejas
- Evitar bloqueos o tiempos de espera excesivos

### `size`

- Indica **cuántos documentos** se devolverán en la respuesta.
    
- No afecta el total de documentos encontrados (`hits.total`).
    
- Por defecto, Elasticsearch retorna **10 documentos**.
    



### `from`

- Se utiliza para **paginación**.
    
- Indica cuántos documentos deben **omitirse** antes de comenzar a devolver resultados.
    
- Combinado con `size`, permite recorrer páginas de resultados.
    

Ejemplo:

- `from = 0`, `size = 10` → primera página
    
- `from = 10`, `size = 10` → segunda página
    

---

### Ejemplo: uso de `size` y `from`

```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "match_all": {}
        },
        "size": 10,
        "from": 10
    }
)

for hit in response["hits"]["hits"]:
    print(hit["_source"])
```

---

### Ejemplo: uso de `timeout`

```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "match": {
                "message": "search keyword"
            }
        },
        "timeout": "10s"
    }
)

response.body
```

Si la búsqueda excede los 10 segundos, Elasticsearch devuelve los resultados parciales disponibles hasta ese momento.

---

### Aggregations

Las **aggregations** permiten realizar **cálculos sobre los datos** sin necesidad de traer todos los documentos.

Algunos ejemplos comunes:

- `avg` → promedio
- `min` → mínimo
- `max` → máximo
- `count` → conteo


Las aggregations no afectan los resultados devueltos en `hits`, sino que se calculan en paralelo.

### Ejemplo: Aggregation de promedio (`avg`)

```python
response = es.search(
    index="my_index",
    body={
        "query": {
            "match_all": {}
        },
        "aggs": {
            "avg_age": {
                "avg": {
                    "field": "age"
                }
            }
        }
    }
)

average_age = response["aggregations"]["avg_age"]["value"]
print(f"Average Age: {average_age}")
```

### Dense Vector

Un **dense vector** es un vector que está mayormente compuesto por valores distintos de cero.
- Almacena vectores densos de valores numéricos.
- Se utiliza cuando el vector tiene pocos o ningún componente en cero.
- No soporta **aggregations** ni **sorting**.
- No es posible almacenar múltiples valores en un mismo campo de tipo `dense_vector`.
- Para la recuperación de información se utiliza **KNN search**, que permite obtener los vectores más cercanos.
- Existe una limitación en la dimensionalidad: el vector puede tener hasta **4096 dimensiones**.
- El mapping debe definirse **manualmente**.
- Elasticsearch **no infiere automáticamente** el mapping para dense vectors.
- Es obligatorio especificar el **número exacto de dimensiones (`dims`)**.
    


### Ejemplo: definición de mapping

Primero se elimina el índice si existe y luego se crea con el mapping manual para el dense vector.

```python
es.indices.delete(index="my_index", ignore_unavailable=True)

es.indices.create(
    index="my_index",
    mappings={
        "properties": {
            "sides_length": {
                "type": "dense_vector",
                "dims": 4
            },
            "shape": {
                "type": "keyword"
            }
        }
    }
)
```

---

### Ejemplo: inserción de documento

```python
from pprint import pprint

response = es.index(
    index="my_index",
    id=1,
    document={
        "shape": "square",
        "sides_length": [5, 5, 5, 5]
    }
)

pprint(response.body)
```


### KNN Search

La búsqueda **KNN (k-nearest neighbors)** se utiliza exclusivamente con campos de tipo `dense_vector`.

- Solo funciona con **dense vectors**.
- No se utiliza `query`; se usa el parámetro `knn`.
- Puede emplearse para **clasificación** y **regresión**.
- El algoritmo compara un nuevo vector con los **k vectores más cercanos** del conjunto de datos.
- Las métricas de distancia más comunes son:
    
    - Euclidean
    - Manhattan        
    - Minkowski

- Elasticsearch permite recuperar hasta **50 candidatos (`num_candidates`)** antes de calcular la distancia final.
- Luego se seleccionan los **k mejores documentos**.
    

### Ejemplo: KNN Search
En este ejemplo, de 5 candidatos se seleccionan los 3 más cercanos.

```python
from pprint import pprint

query = "What is a black hole?"
embedded_query = get_embedding(query)

result = es.search(
    index="my_index",
    knn={
        "field": "embedding",
        "query_vector": embedded_query,
        "num_candidates": 5,
        "k": 3
    }
)

n_documents = result.body["hits"]["total"]["value"]
print(f"Found {n_documents} documents")

hits = result.body["hits"]["hits"]
for hit in hits:
    print(f"Title  : {hit['_source']['title']}")
    print(f"Content: {hit['_source']['content']}")
    print(f"Score  : {hit['_score']}")
    print("*" * 100)
```

