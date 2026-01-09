
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
    



### Search API
- Podes buscar en varios index separados por comas  o buscar en todos  _all
- q para queries simples
- query es para complejos queires y usa query dsl language
- timeout( el tiempo maximo  de espera), size(elnumero de documentos), from(starting point pagination)


