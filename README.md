# 🗄️ DataChat · Consulta de Bases de Datos mediante Agentes Autónomos y LLMs

> **Asignatura:** Programación de Bases de Datos (PBD)  
> **Repositorio:** [github.com/elenacanchog/proyecto-llm-bbdd](https://github.com/elenacanchog/proyecto-llm-bbdd)

**DataChat** es un sistema inteligente que permite interrogar bases de datos **Relacionales (SQL)**, **No Relacionales (NoSQL)** y **archivos locales (`.csv` / `.json`)** utilizando **lenguaje natural en español**.

El proyecto aborda uno de los retos principales de la integración entre Modelos de Lenguaje (LLMs) y sistemas de almacenamiento: **¿cómo lograr que un modelo genere consultas exactas sin conocer previamente la estructura de los datos y sin inventarse nombres de tablas o campos?** Para resolverlo, se ha diseñado una arquitectura basada en **Agentes Autónomos con herramientas (*Tool Calling*)** y se ha realizado una comparativa experimental entre **inferencia en la nube (APIs externas)** e **inferencia 100 % local**.

---

## 🧠 ¿Cómo funciona el sistema por dentro?

A diferencia de los enfoques tradicionales de *Text-to-SQL* (donde se le pasa un esquema estático al modelo en un único *prompt* y se cruza los dedos para que acierte), este proyecto implementa **Agentes ReAct (Razonamiento + Acción)** mediante **LangChain**.

Cuando el usuario formula una pregunta en español, el agente sigue un ciclo autónomo de decisión:

```text
Pregunta en español
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ 1. INSPECCIÓN DEL ESQUEMA (Tool: get_schema)             │
│    El agente consulta qué tablas o colecciones existen,  │
│    sus tipos de datos y ejemplos reales de valores.      │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│ 2. GENERACIÓN Y EJECUCIÓN (Tool: run_sql / run_mongo)    │
│    El agente redacta la consulta SQL o el Pipeline de    │
│    agregación MongoDB y lo ejecuta contra el motor real. │
└──────────────────────────┬───────────────────────────────┘
                           │
             ¿Error de sintaxis o 0 filas?
                ├── SÍ ──► 3. AUTOCORRECCIÓN (Reintenta ajustando campos/tipos)
                └── NO ──► 4. SÍNTESIS Y RESPUESTA FINAL EN ESPAÑOL
```

### Optimizaciones técnicas implementadas
* **Compactación inteligente de esquemas NoSQL (`resumir_valor`):** Los documentos de MongoDB suelen tener estructuras profundamente anidadas y listas largas. Antes de entregar un documento de ejemplo al LLM, una función recursiva acorta cadenas largas y limita las muestras de arrays a 12 elementos, garantizando que el modelo vea todos los campos reales sin desbordar su ventana de contexto.
* **Proyección negativa de campos pesados:** En MongoDB Atlas se excluyen automáticamente campos textuales masivos (`reviews`, `description`, URLs de imágenes, etc.) para evitar saturar el límite de tokens por minuto de las APIs gratuitas.
* **Soporte de archivos propios al vuelo (en memoria RAM):** En la aplicación web, si el usuario sube un `.csv`, el sistema levanta una base de datos **SQLite** temporal en memoria; si sube un `.json`, levanta un servidor **MongoDB** simulado en memoria mediante `mongomock`.

---

## ⚖️ Arquitectura Híbrida: Nube vs. Local

El proyecto permite ejecutar y comparar dos paradigmas distintos de despliegue de LLMs:

| Característica | ☁️ Modo API en la Nube (`api_cloud`) | 🖥️ Modo Local (`local_llm`) |
| :--- | :--- | :--- |
| **Motor de Inferencia** | **Groq Cloud API** (aceleración LPU) | **Ollama** (ejecución local en CPU/GPU propia) |
| **Modelos Evaluados** | `openai/gpt-oss-120b`<br>`qwen/qwen3.8-27b`<br>`openai/gpt-oss-20b` | `qwen3.5:4b`<br>`gemma-4:latest` |
| **Base de Datos SQL** | **PostgreSQL** alojado en **Neon** | **PostgreSQL** local (`localhost:5432`) |
| **Base de Datos NoSQL** | **MongoDB Atlas** (clúster en la nube) | **MongoDB Community** local (`localhost:27017`) |
| **Privacidad y Coste** | Muy rápido; sujeto a cuotas de tokens/minuto | 100 % privado y sin internet; depende del hardware |
| **Modo de Evaluación** | Batería automatizada + métricas + gráficos PNG | Visualización en tiempo real (*streaming*) en consola |

### Datasets utilizados
1. **Base de datos Relacional (`nasa_meteoritos`):** Dataset oficial de la NASA (*Meteorite Landings*) con **45.716 registros** sobre caídas de meteoritos (nombre, masa en gramos, clase, año, coordenadas). Ideal para consultas analíticas con agrupaciones, filtros numéricos y manejo de valores nulos (`NULLS LAST`).
2. **Base de datos No Relacional (`sample_airbnb`):** Colección `listingsAndReviews` con **5.555 alojamientos turísticos** en formato JSON documental (subdocumentos de dirección, anfitrión, puntuaciones y arrays de servicios/comodidades). Ideal para evaluar *Aggregation Pipelines* (`$match`, `$unwind`, `$group`, `$sort`, `$project`).

---

## 📁 Estructura del Proyecto y Explicación de Archivos

```text
proyecto-llm-bbdd/
├── demo_web.py                      # Aplicación web interactiva (DataChat) en Streamlit
├── requirements.txt                 # Lista de dependencias Python del proyecto
├── .env.example                     # Plantilla pública de variables de entorno
├── .gitignore                       # Archivos excluidos de Git (.env, venv, datasets pesados)
├── presentacion.html                # Material de apoyo visual para la exposición del proyecto
├── data/                            # Directorio destinado a los datasets originales
│   └── .gitkeep                     # Mantiene la carpeta visible en Git aunque esté vacía
├── graficos/                        # Gráficas comparativas generadas por los scripts de nube
│   ├── comparativa_tiempo_mongo_api.png
│   ├── comparativa_tiempo_sql_api.png
│   ├── comparativa_tokens_mongo_api.png
│   └── comparativa_tokens_sql_api.png
└── src/
    ├── api_cloud/                   # MÓDULO NUBE (Groq + MongoDB Atlas + PostgreSQL Neon)
    │   ├── cargar_csv_cloud.py      # Script ETL de limpieza y carga inicial hacia Neon
    │   ├── sql_api.py               # Agente SQL + batería de pruebas, telemetría y gráficos
    │   └── mongo_api.py             # Agente NoSQL + batería de pruebas, telemetría y gráficos
    └── local_llm/                   # MÓDULO LOCAL (Ollama + PostgreSQL y MongoDB locales)
        ├── cargar_csv.py            # Script ETL de carga del CSV hacia PostgreSQL local
        ├── cargar_json.py           # Script ETL de carga del JSON hacia MongoDB local
        ├── sql_local.py             # Agente SQL local con salida en streaming en vivo
        └── mongo_local.py           # Agente NoSQL local con salida en streaming en vivo
```

### Descripción detallada de cada archivo

* **Raíz del proyecto:**
  * `demo_web.py`: Interfaz gráfica web desarrollada con **Streamlit**. Permite alternar entre Nube y Local, elegir entre MongoDB, PostgreSQL o subir archivos propios (`.csv`/`.json`), probar preguntas de ejemplo, inspeccionar la consulta exacta generada por el agente y consultar métricas de latencia y consumo de tokens.
  * `.env.example`: Plantilla con los nombres de las variables necesarias para conectar con las APIs y bases de datos sin exponer contraseñas reales en el repositorio.
  * `requirements.txt`: Archivo de dependencias para reproducir exactamente el entorno Python en cualquier equipo.
* **Carpeta `src/api_cloud/` (Experimentación en la Nube):**
  * `cargar_csv_cloud.py`: Lee `data/Meteorite_Landings.csv`, limpia los tipos de datos (convierte años y masas a formato numérico) y crea/puebla la tabla en PostgreSQL Neon.
  * `sql_api.py` y `mongo_api.py`: Ejecutan una batería estandarizada de preguntas de dificultad creciente sobre los tres modelos de Groq. Recogen métricas de éxito, número de reintentos de autocorrección, tiempo total (s) y tokens consumidos, imprimen una tabla comparativa con `pandas` y exportan automáticamente las gráficas de barras en la carpeta `graficos/`.
* **Carpeta `src/local_llm/` (Experimentación 100 % Local):**
  * `cargar_csv.py` y `cargar_json.py`: Scripts de ingesta que cargan los datasets de `data/` en las instancias locales de PostgreSQL y MongoDB respectivamente.
  * `sql_local.py` y `mongo_local.py`: Ejecutan los agentes utilizando `ChatOllama` sobre modelos locales. Incorporan un indicador visual multihilo (`Spinner`) y utilizan `agente.stream()` para mostrar paso a paso en tiempo real cómo razona el modelo y qué herramientas invoca.

---

## 🛠️ Guía de Instalación Paso a Paso

### Requisitos previos
* **Python 3.10 o superior** instalado en el sistema.
* **Git** para clonar el repositorio.
* *(Opcional, solo para el modo local)*: Tener instalados **Ollama**, **PostgreSQL** y **MongoDB Community Server**.

### Paso 1: Clonar el repositorio y crear el entorno virtual
Abre una terminal y ejecuta los siguientes comandos:

```bash
# 1. Clonar el proyecto
git clone [https://github.com/elenacanchog/proyecto-llm-bbdd.git](https://github.com/elenacanchog/proyecto-llm-bbdd.git)
cd proyecto-llm-bbdd

# 2. Crear el entorno virtual aislado
python3 -m venv venv_proyectopbd

# 3. Activar el entorno virtual
source venv_proyectopbd/bin/activate      # En Linux / macOS
# venv_proyectopbd\Scripts\activate       # En Windows (PowerShell/CMD)

# 4. Instalar todas las librerías necesarias
pip install -r requirements.txt
```

### Paso 2: Descargar los datasets originales
Debido a que el archivo JSON supera el límite de tamaño permitido por GitHub (`>100 MB`), los datasets no están alojados directamente en el control de versiones.

1. Descarga ambos archivos desde la carpeta compartida del proyecto:  
   📥 **[Descargar datasets (`Meteorite_Landings.csv` y `sample_airbnb.json`)](https://unexes-my.sharepoint.com/:f:/g/personal/rrinamen_alumnos_unex_es/IgDeoFLlgqVwTLxO8F1TJMtiAT9tHBqa3mb7MWTNleMfd4E?e=AOcaHv)**
2. Coloca los dos archivos descargados directamente dentro de la carpeta **`data/`** del proyecto:
   * `data/Meteorite_Landings.csv`
   * `data/sample_airbnb.json`

### Paso 3: Configurar las variables de entorno (`.env`)
Copia la plantilla `.env.example` para crear tu archivo privado `.env`:

```bash
cp .env.example .env
```

Abre el archivo `.env` con cualquier editor de texto y rellena los datos según el modo que vayas a utilizar:

```env
# =============================================================================
# 1. CREDENCIALES MODO NUBE (Necesarias para api_cloud y demo_web en modo Nube)
# =============================================================================
MONGODB_USERNAME=tu_usuario_atlas
MONGODB_PASSWORD=tu_password_atlas
MONGODB_URI=mongodb+srv://usuario:password@cluster.mongodb.net/
GROQ_API_KEY=gsk_tu_clave_gratuita_de_groq
POSTGRES_URI=postgresql+psycopg2://usuario:password@host.neon.tech/nasa_meteoritos?sslmode=require

# =============================================================================
# 2. CONFIGURACIÓN MODO LOCAL (Necesarias para local_llm y demo_web en modo Local)
# =============================================================================
LOCAL_MONGO_URI=mongodb://localhost:27017/
LOCAL_MONGO_DB=sampl_airbnb
LOCAL_POSTGRES_URI=postgresql+psycopg2://postgres:tu_password@localhost:5432/nasa_meteoritos
```

* **¿Dónde obtener las credenciales de la nube gratuitas?**
  * `GROQ_API_KEY`: Creando una cuenta gratuita en [console.groq.com](https://console.groq.com/).
  * `MONGODB_URI`: Creando un clúster gratuito M0 en [MongoDB Atlas](https://www.mongodb.com/atlas) y activando el dataset de ejemplo *Sample Dataset* (`sample_airbnb`).
  * `POSTGRES_URI`: Creando un proyecto gratuito en [Neon.tech](https://neon.tech/) con una base de datos llamada `nasa_meteoritos`.

---

## ▶️ Guía de Ejecución

### Escenario A: Arrancar la Interfaz Web Interactiva (`DataChat`)
Es la forma más rápida de probar todas las funcionalidades del sistema desde el navegador:

```bash
streamlit run demo_web.py
```
Una vez abierta la pestaña en el navegador (`http://localhost:8501`):
1. Selecciona el **Entorno de ejecución** (*API en la Nube* o *Local*).
2. Elige la **Fuente de datos** (*MongoDB*, *PostgreSQL* o *Archivo propio `.CSV` / `.JSON`*).
3. Selecciona el **Modelo LLM** que quieras poner a prueba.
4. Escribe tu pregunta en español (o pulsa uno de los botones de ejemplo) y haz clic en **🚀 Ejecutar Consulta**.

---

### Escenario B: Ejecutar los scripts de evaluación en la Nube (`api_cloud`)
1. **Poblar PostgreSQL en Neon** (solo es necesario hacerlo la primera vez):
   ```bash
   python3 src/api_cloud/cargar_csv_cloud.py
   ```
2. **Lanzar las baterías comparativas:**
   ```bash
   # Evaluación sobre PostgreSQL (Neon)
   python3 src/api_cloud/sql_api.py

   # Evaluación sobre MongoDB Atlas
   python3 src/api_cloud/mongo_api.py
   ```
   Al finalizar cada script, se mostrará en consola la tabla resumen con los tiempos y tokens de cada modelo y se actualizarán las imágenes `.png` dentro de la carpeta `graficos/`.

---

### Escenario C: Ejecutar el entorno 100 % Local (`local_llm`)
1. **Descargar los modelos en Ollama:**
   Asegúrate de tener Ollama ejecutándose en segundo plano (`ollama serve`) y descarga los modelos configurados:
   ```bash
   ollama pull qwen3.5:4b
   ollama pull gemma-4:latest
   ```
2. **Poblar las bases de datos locales** (solo la primera vez):
   Con los servicios locales de PostgreSQL y MongoDB activos:
   ```bash
   python3 src/local_llm/cargar_csv.py
   python3 src/local_llm/cargar_json.py
   ```
3. **Ejecutar los agentes locales en tiempo real:**
   ```bash
   python3 src/local_llm/sql_local.py
   python3 src/local_llm/mongo_local.py
   ```

---

## 🔧 Solución de Problemas Frecuentes

* **`Error 429 (Rate Limit Exceeded)` en modo Nube:** La capa gratuita de Groq limita el número de tokens procesados por minuto. Si lanzas muchas consultas seguidas con modelos grandes (`gpt-oss-120b`), espera 30–60 segundos o cambia temporalmente al modelo `openai/gpt-oss-20b`.
* **`Connection refused` en modo Local:** Verifica que el servicio de Ollama (`ollama serve`), PostgreSQL (`5432`) y MongoDB (`27017`) están arrancados en tu máquina y que las credenciales de `LOCAL_POSTGRES_URI` en tu `.env` coinciden con tu contraseña local.
* **Bloqueo de puertos en redes universitarias/corporativas:** Algunas redes Wi-Fi bloquean los puertos salientes de bases de datos (`5432` para PostgreSQL o `27017` para MongoDB Atlas). Si ocurre un *timeout* al conectar con Neon o Atlas, prueba desde una red doméstica o compartiendo datos móviles.