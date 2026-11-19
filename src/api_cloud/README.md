# Acceso a Bases de Datos No Relacionales (MongoDB Atlas) mediante LLMs vía API

Este módulo implementa una arquitectura **Text-to-NoSQL** utilizando **LangChain**, la API de **Groq** y **MongoDB Atlas**. El sistema permite consultar en lenguaje natural (español) una base de datos documental compleja (`sample_airbnb`), traduciendo las preguntas del usuario a *Pipelines de Agregación* ejecutables en MongoDB, recuperando los datos en tiempo real y midiendo el rendimiento comparativo de distintos modelos de lenguaje.

---

## 1. Fundamento Técnico: ¿Cómo accede un LLM a una Base de Datos?

Un modelo de lenguaje (LLM) **no puede ni debe recibir toda la base de datos en su prompt**: los 5.555 documentos de `sample_airbnb` superarían el límite de ventana de contexto, dispararían el coste de tokens y comprometerían la privacidad de los datos.

En su lugar, este sistema utiliza al LLM como un **compilador semántico** mediante un flujo desacoplado en tres etapas:

1. **Inyección de Esquema (*Schema Grounding*):** Se entrega al modelo únicamente la estructura de los campos disponibles (`ESQUEMA_AIRBNB`), nunca los datos reales.
2. **Traducción a Código de Consulta (*Text-to-MQL*):** El LLM interpreta la intención del usuario y genera una consulta estructurada en formato JSON (*Aggregation Pipeline*).
3. **Ejecución Delegada y Síntesis:** Python envía esa consulta al motor de MongoDB Atlas, que filtra y calcula los resultados en sus servidores en milisegundos. Finalmente, solo los registros resultantes se devuelven al LLM para que redacte una respuesta comprensible en español.

---

## 2. Requisitos y Configuración del Entorno

### A. Entorno Virtual e Instalación
```bash
python3 -m venv venv_proyectopbd
source venv_proyectopbd/bin/activate
pip install langchain langchain-groq langchain-core pymongo python-dotenv pandas dnspython

```

### B. Variables de Entorno (`.env`)

En la raíz del repositorio debe existir un archivo `.env` (excluido del control de versiones en `.gitignore`) con las credenciales:

```env
MONGODB_URI="mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
GROQ_API_KEY="gsk_tu_clave_de_groq"
GROQ_MODEL="openai/gpt-oss-120b"

```

---

## 3. Análisis Detallado de la Función Principal: `consultar_airbnb_con_metricas`

La función `consultar_airbnb_con_metricas(pregunta_usuario: str, modelo: str) -> dict` constituye el núcleo del sistema. A continuación se detalla qué ocurre internamente en cada una de sus fases:

### Paso 0: Inicialización de Telemetría y Configuración Determinista

* **Medición de tiempos (`time.time()`):** Se registra `inicio_total` al arrancar la función y `inicio_llm_1` justo antes de llamar al modelo. Esto permite aislar cuánto tarda el LLM en "pensar" la consulta (`Tiempo_LLM_Query`) frente al tiempo total de red, ejecución en base de datos y redacción final (`Tiempo_Total`).
* **Conexión a clientes:** Se instancia `MongoClient(mongo_uri)` apuntando a la colección `listingsAndReviews` de `sample_airbnb` y `ChatGroq(model=modelo, temperature=0)`.
* **¿Por qué `temperature=0`?** En tareas de generación de código y consultas estructuradas, la creatividad del modelo es perjudicial. Fijar la temperatura a `0` fuerza una selección determinista de tokens, minimizando el riesgo de que el modelo invente nombres de campos o rompa la sintaxis JSON.

### Paso 1: Generación de la Consulta MongoDB (Fase 1 del LLM)

En esta fase se construye la primera cadena de LangChain mediante **LCEL (*LangChain Expression Language*)**: `cadena_query = prompt_generador | llm`.

* **Ingeniería del Prompt (`ChatPromptTemplate`):** El mensaje `system` impone tres restricciones críticas:
1. **Salida JSON estricta:** Obliga a devolver únicamente el array JSON del *Aggregation Pipeline* sin texto explicativo ni bloques Markdown (````json`). Como medida defensiva adicional, el código limpia manualmente posibles etiquetas Markdown antes de parsear.
2. **Optimización mediante `$project` y `$limit`:** Los documentos de Airbnb contienen listas extensas de reseñas (`reviews`) y URLs. Exigir una etapa `$project` (excluyendo `_id`) y un `$limit` evita transferir documentos pesados y previene superar el límite de tokens en la fase final.
3. **Alineación semántica e idiomática:** Aunque el usuario pregunte en español (*"aire acondicionado"*, *"España"*), los valores almacenados en la base de datos están en inglés (`"Air conditioning"`, `"Spain"`). El prompt instruye al modelo para traducir los valores de filtrado manteniendo intactos los nombres técnicos de los campos del esquema.


* **¿Por qué *Aggregation Pipeline* (`aggregate`) y no `find`?** Un pipeline es una tubería secuencial de etapas (`$match` $\rightarrow$ `$group` $\rightarrow$ `$sort` $\rightarrow$ `$project` $\rightarrow$ `$limit`). Permite resolver en una única estructura JSON tanto búsquedas sencillas con filtros anidados como cálculos estadísticos complejos (medias con `$avg` y agrupaciones con `$group`).
* **Captura de Tokens:** Al invocar `cadena_query.invoke(...)`, LangChain devuelve un objeto `AIMessage`. De su diccionario `response_metadata["token_usage"]` se extrae `total_tokens` para contabilizar el gasto exacto de esta primera llamada.

### Paso 2: Validación JSON, Ejecución en Atlas y Puente BSON $\rightarrow$ JSON

Una vez recibido el texto con la consulta generada por el LLM:

1. **Deserialización (`json.loads(texto_pipeline)`):** Convierte la cadena de texto en una lista real de diccionarios de Python. Si el modelo cometió un error de sintaxis JSON, el bloque `try/except` captura la excepción, registra el fallo en `metricas["Error"]` y detiene la ejecución de forma controlada sin romper el programa.
2. **Ejecución Remota (`coleccion.aggregate(pipeline)`):** PyMongo envía el pipeline a los servidores de MongoDB Atlas. El motor NoSQL ejecuta el filtrado y las agrupaciones en la nube y devuelve únicamente los registros coincidentes.
3. **Serialización BSON (`bson.json_util.dumps`):** MongoDB almacena precios y cifras exactas en formato binario `Decimal128`, un tipo de dato que la librería estándar `json` de Python no sabe serializar. `json_util.dumps(..., ensure_ascii=False)` traduce de forma segura los tipos nativos de MongoDB (BSON) a texto JSON legible manteniendo tildes y caracteres especiales.

### Paso 3: Redacción de Respuesta Fundamentada (Fase 2 del LLM)

Los datos devueltos por Atlas están en formato JSON crudo. Para presentarlos de forma comprensible, se ejecuta una **segunda cadena de LangChain** (`cadena_final = prompt_respuesta | llm`):

* **Control de Alucinaciones (*Grounding*):** El mensaje de sistema ordena al LLM responder en español basándose **exclusivamente** en el bloque `{datos}` inyectado. Si la base de datos devolvió una lista vacía (`[]`), el modelo tiene orden explícita de informar que no existen resultados, impidiendo que invente alojamientos utilizando su memoria de entrenamiento.
* **Cierre de Métricas:** Se suman los tokens de esta segunda llamada (`tokens_2`) a los de la primera (`tokens_1`), se calcula el `Tiempo_Total (s)` y se retorna el diccionario `metricas`.

---

## 4. Estructura Completa del Archivo (`src/api_cloud/mongo_api.py`)

| Componente | Tipo | Función en el sistema |
| --- | --- | --- |
| `ESQUEMA_AIRBNB` | Constante (`str`) | Define la estructura de `listingsAndReviews` (campos raíz, listas como `amenities` y subdocumentos como `address` y `host`). |
| `probar_conexion_airbnb()` | Función auxiliar | Verifica la conectividad TLS con Atlas, cuenta los `5.555` documentos y muestra un registro de prueba sin usar IA. |
| `consultar_airbnb_con_ia()` | Función base | Ejecuta el flujo completo de consulta en lenguaje natural para una única pregunta, sin recolección de métricas. |
| `imprimir_banner_modelo()` / `imprimir_banner_pregunta()` | Utilidades visuales | Formatean la salida en consola con separadores claros para distinguir cada modelo y pregunta durante las baterías de prueba. |
| `consultar_airbnb_con_metricas()` | Motor principal | Implementa las 3 fases (Text-to-MQL, ejecución en Atlas y síntesis) registrando tiempos, éxito sintáctico y consumo de tokens. |
| Bloque `__main__` | Banco de pruebas | Itera sobre las listas configurables `MIS_MODELOS` y `MIS_PREGUNTAS`, aplica pausas preventivas (`time.sleep(2)`) para respetar los límites de la API gratuita y consolida todos los resultados en un `pandas.DataFrame` comparativo. |

---

## 5. Ejecución

Con el entorno virtual activado y desde la raíz del proyecto:

```bash
python3 src/api_cloud/mongo_api.py

```

