# Guion de presentación — AgenteSQL


## 0. Apertura (0:00 – 0:30)

**Qué se ve:** portada / slide con el título del proyecto y el esquema horizontal del flujo.

**Qué se dice:**
> "Buenos días. Este proyecto trata de una pregunta sencilla: ¿qué pasaría si pudieras hablar con una base de datos como si le hablaras a un compañero? No con SQL, no con filtros, sino en español. El sistema se llama AgenteSQL, y en los próximos diez minutos os voy a enseñar cómo lo construimos, cómo accede a los datos, cómo corre en tu portátil y cómo corre en la nube, y lo cerraremos con una demo en vivo y una comparativa de resultados."

**Visual:** título + esquema del flujo del agente (pregunta -> procesa -> get_schema -> consulta -> ejecuta -> responde).

---

## 1. LangChain: qué es, para qué sirve y qué hace en nuestro proyecto (0:30 – 2:30)

### 1.1 Definición (0:30 – 1:00)

**Qué se dice:**
> "LangChain es un *framework* de código abierto para construir aplicaciones sobre modelos de lenguaje. No es un modelo ni una base de datos: es una capa de orquestación. Su trabajo es conectar el modelo con el resto del sistema: los *prompts*, las herramientas externas, la memoria, el streaming y el formato de salida. En nuestro proyecto usamos la versión 0.3, con los paquetes `langchain-core`, `langchain-community`, `langchain-groq` y `langchain-ollama`."

**Visual:** logo de LangChain + lista de paquetes usados.

### 1.2 Para qué sirve (1:00 – 1:30)

**Qué se dice:**
> "Sirve para tres cosas que nos importan. Primera: **composición** — encadenar pasos (prompt, modelo, parser) con un operador simple. Segunda: **herramientas** — registrar funciones de Python como tools que el modelo puede invocar por su cuenta. Tercera: **agentes** — un bucle en el que el modelo razona, actúa, observa el resultado y decide si necesita otro paso. Sin LangChain, todo eso habría que programarlo a mano contra la API de cada proveedor."

**Visual:** diagrama de los tres bloques: Composición · Herramientas · Agentes.

### 1.3 Qué hace en nuestro proyecto (1:30 – 2:10)

**Qué se dice:**
> "En AgenteSQL, LangChain es el director de orquesta. Define los *prompts* del sistema, registra las tools `get_schema`, `run_sql` y `run_mongo`, monta el agente ReAct y se encarga del streaming de la respuesta. También nos permite cambiar de modelo —Groq en la nube o Ollama en local— tocando una sola línea, porque todos comparten la misma interfaz `BaseChatModel`."

### 1.4 El flujo del agente (2:10 – 2:30)

**Qué se dice (recorriendo el esquema):**
> "Cuando el usuario pregunta en español, el agente sigue este ciclo de forma autónoma:
>
> 1. **Pregunta del humano** — en lenguaje natural.
> 2. **El agente procesa la pregunta** — razona sobre lo que se le pide.
> 3. **Llama a la tool del esquema** (`get_schema`) — consulta qué tablas o colecciones existen, sus tipos y ejemplos reales de valores.
> 4. **Crea la consulta** — redacta el SQL para PostgreSQL o el *aggregation pipeline* para MongoDB.
> 5. **Llama a la tool de ejecutar** (`run_sql` / `run_mongo`) — lanza la consulta contra el motor real.
> 6. **Interpreta los datos y los devuelve formateados** al usuario en español.
>
> Si la ejecución da error de sintaxis o devuelve cero filas, el agente no se rinde: corrige la consulta y reintenta. Eso es el bucle de autocorrección."

**Visual:** esquema horizontal del flujo (el SVG `assets/flujo-agente.svg`).

---

## 2. Las bases de datos: cómo accedemos, filas, estructuras y columnas (2:30 – 4:00)

### 2.1 Por qué dos bases distintas (2:30 – 2:50)

**Qué se dice:**
> "Usamos dos bases para cubrir los dos mundos que el proyecto promete: relacional y documental. Una no sustituye a la otra; sirven para problemas distintos."

### 2.2 PostgreSQL — relacional (2:50 – 3:20)

**Qué se dice:**
> "Dataset: *Meteorite Landings* de la NASA, 45.716 registros. Lo dejamos en una sola tabla llamada `meteoritos` con columnas tipadas: `name` (texto), `mass_g` (numérico), `year` (entero, admite nulos), `recclass` (clase del meteorito), `fall` (caído o encontrado), `reclat` y `reclong` (coordenadas). Cada fila es un meteorito; cada columna es un atributo. El CSV se limpia con pandas —se descarta `GeoLocation`, se renombra `mass (g)` a `mass_g`, y `year` pasa a entero nullable— y SQLAlchemy crea la tabla con tipos explícitos. Así el esquema que lee el agente es fiable, no una suposición."

**Visual:** tabla `meteoritos` con 4-5 filas de ejemplo + tipos.

### 2.3 MongoDB — documental (3:20 – 3:45)

**Qué se dice:**
> "Dataset: `sample_airbnb`, colección `listingsAndReviews`, 5.555 documentos. Aquí no hay tabla ni columnas fijas: cada documento es un JSON con subdocumentos (`address`, `host`, `review_scores`) y arrays (`amenities`, `reviews`). Los campos pueden variar de un documento a otro. El agente no recibe los documentos: recibe una descripción de la estructura y un ejemplo compactado."

**Visual:** un documento JSON de ejemplo indentado.

### 2.4 Cómo accedemos (3:45 – 4:00)

**Qué se dice:**
> "El acceso es idéntico en los dos casos: el agente invoca la tool, esta se conecta con el driver correspondiente —`psycopg2` para PostgreSQL, `pymongo` para MongoDB—, ejecuta la consulta y devuelve filas o documentos. El modelo nunca toca la base directamente; solo ve el resultado."

---

## 3. Ollama y la IA en local (4:00 – 5:30)

### 3.1 Qué es Ollama (4:00 – 4:30)

**Qué se dice:**
> "Ollama es una herramienta que descarga, instala y ejecuta LLMs en tu propia máquina. Expone una API local compatible con OpenAI, así que LangChain se conecta a ella con `ChatOllama` como si fuera un servicio remoto. En nuestro proyecto usamos `qwen3.5:4b` y `gemma-4`."

### 3.2 Por qué nos interesa (4:30 – 5:00)

**Qué se dice:**
> "Tres motivos. Privacidad: el prompt, los datos y la respuesta no salen del equipo. Coste: no se paga por token. Y control: puedes ajustar el contexto (`num_ctx = 8192`), mantener el modelo cargado en memoria (`keep_alive = 15m`) y fijar la temperatura a cero para reproducibilidad. El precio se paga en hardware y en velocidad: un modelo pequeño razona peor y más despacio que uno grande."

### 3.3 Cómo lo lanzamos (5:00 – 5:30)

**Qué se dice:**
> "Se arranca el servicio con `ollama serve`, se descarga el modelo con `ollama pull gemma-4`, y el agente local ya está listo. La base de datos también es local: PostgreSQL en el puerto 5432 y MongoDB en el 27017. Todo el flujo se imprime en la consola en tiempo real, paso a paso, incluyendo cada tool que invoca el agente."

**Visual:** terminal con el streaming del agente local (captura previa).

---

## 4. La nube: LLM por API (5:30 – 7:00)

### 4.1 Qué es un LLM por API (5:30 – 5:55)

**Qué se dice:**
> "En la nube no ejecutamos el modelo: lo llamamos por HTTP. Un proveedor aloja el modelo en su hardware y nos da un *endpoint* con una clave. Nosotros enviamos el prompt y recibimos la respuesta. En nuestro caso el proveedor es Groq, que acelera modelos abiertos con su hardware propio, y la integración con LangChain se hace con `ChatGroq`."

### 4.2 Modelos evaluados (5:55 – 6:20)

**Qué se dice:**
> "Evaluamos tres modelos de Groq: `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` y `openai/gpt-oss-20b`. Las bases también están en la nube: PostgreSQL en Neon y MongoDB Atlas con el dataset de ejemplo `sample_airbnb`."

### 4.3 Ventajas y contrapartidas (6:20 – 6:45)

**Qué se dice:**
> "Ventajas: latencia baja, calidad alta y cero mantenimiento de infraestructura. Contrapartidas: los datos salen de la máquina —una decisión de privacidad—, hay coste por token y límites de cuota. Por eso la temperatura está a cero: queremos reproducibilidad, no sorpresas, y cada token cuenta."

### 4.4 Dónde vive la clave (6:45 – 7:00)

**Qué se dice:**
> "Las credenciales viven en un archivo `.env` (plantilla en `.env.example`): `GROQ_API_KEY`, `MONGODB_URI`, `POSTGRES_URI`. El código las lee con `python-dotenv` y nunca se suben al repositorio."

---

## 5. Streamlit: la demo (7:00 – 9:00)

> **Preparación previa:** arrancar `streamlit run demo_web.py` antes de la presentación y dejarla abierta en `http://localhost:8501`. También tener la CLI local lista por si se quiere mostrar el streaming en consola.

### 5.1 Qué es Streamlit (7:00 – 7:20)

**Qué se dice:**
> "Streamlit es una librería de Python que convierte un script en una aplicación web interactiva sin escribir HTML ni JavaScript. Cada widget (selector, botón, caja de texto) se declara en Python y Streamlit se encarga de la interfaz. Es ideal para prototipos de ciencia de datos y, en nuestro caso, para enseñar el agente."

### 5.2 Qué hace nuestra demo (7:20 – 7:50)

**Qué se dice:**
> "La demo web permite alternar entre nube y local, elegir entre MongoDB, PostgreSQL o subir un archivo propio (`.csv`/`.json`), seleccionar el modelo LLM, escribir una pregunta en español o pulsar un botón de ejemplo, y al ejecutar muestra: la respuesta en español, la consulta exacta que generó el agente, los datos crudos y las métricas de latencia y tokens."

### 5.3 Demo en vivo (7:50 – 8:40)

**Qué se hace:**
1. Seleccionar **API en la nube** + **MongoDB** + modelo `gpt-oss-120b`.
2. Pregunta de ejemplo: *"¿Qué alojamientos tienen wifi y aire acondicionado en Barcelona?"*
3. Ejecutar y comentar la respuesta.
4. Abrir el desplegable de la consulta para enseñar el *aggregation pipeline* generado.
5. Cambiar a **PostgreSQL** y preguntar: *"Los 3 meteoritos más pesados caídos después de 1900"*.
6. (Opcional) Mostrar la CLI local con `python src/local_llm/sql_local.py` para enseñar el streaming paso a paso.

**Qué se dice durante la demo:**
> "Fijaos en que no hemos escrito SQL ni MQL. El agente ha leído el esquema, ha generado la consulta, la ha ejecutado y ha formateado el resultado. Y si algo falla, reintenta solo."

### 5.4 Archivos propios (8:40 – 9:00)

**Qué se dice:**
> "Un detalle: si subes un `.csv`, la demo levanta una base SQLite temporal en memoria; si subes un `.json`, levanta un servidor MongoDB simulado con `mongomock`. Así cualquier usuario puede interrogar sus propios datos sin infraestructura."

---

## 6. Conclusiones y tablas comparativas (9:00 – 10:00)

> **Nota:** esta sección se completa una vez cerrada la batería de pruebas. Aquí se deja el esqueleto y los puntos a cubrir.

### 6.1 Puntos a tratar (9:00 – 9:20)

- **Local vs nube:** privacidad y coste frente a velocidad y calidad.
- **SQL vs NoSQL:** el esquema relacional es más predecible para el modelo; el documental exige compactar y proyectar.
- **Modelos:** los más grandes son más fiables generando consultas válidas; los pequeños son más rápidos pero fallan más.
- **Reintentos:** la autocorrección reduce los fallos, pero consume tokens y tiempo.
- **Aprendizaje clave:** el límite no es el modelo, es el contexto. Menos datos en el prompt, más consultas correctas.

### 6.2 Tabla comparativa de modelos (9:20 – 9:45)

| Modelo | Entorno | Latencia consulta (s) | Latencia total (s) | Tokens | Éxito |
|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | Groq (nube) | — | — | — | — |
| `qwen/qwen3.8-27b` | Groq (nube) | — | — | — | — |
| `openai/gpt-oss-20b` | Groq (nube) | — | — | — | — |
| `qwen3.5:4b` | Ollama (local) | — | — | — | — |
| `gemma-4` | Ollama (local) | — | — | — | — |

> Rellenar tras ejecutar `sql_api.py` y `mongo_api.py` (nube) y los scripts locales.

### 6.3 Tabla comparativa de entornos (9:45 – 9:55)

| Característica | Modo nube (`api_cloud`) | Modo local (`local_llm`) |
|---|---|---|
| Motor de inferencia | Groq (LPU) | Ollama (CPU/GPU propia) |
| Base SQL | PostgreSQL en Neon | PostgreSQL local (`localhost:5432`) |
| Base NoSQL | MongoDB Atlas | MongoDB Community local (`localhost:27017`) |
| Privacidad | Los datos salen de la máquina | 100 % privado, sin internet |
| Coste | Por token (cuota gratuita) | Hardware propio |
| Evaluación | Batería automatizada + gráficos PNG | Streaming en tiempo real en consola |

### 6.4 Cierre (9:55 – 10:00)

**Qué se dice:**
> "En resumen: no hace falta que el modelo sepa la base de datos. Hace falta que sepa preguntarle. LangChain nos da la orquestación, Ollama y Groq nos dan el razonamiento, y las bases de datos hacen lo que mejor saben: ejecutar consultas exactas. Muchas gracias."

**Visual:** slide de cierre con el esquema del flujo y la frase *"El modelo no lee la base de datos: aprende a preguntarle."*

---

## Apéndice — checklist pre-presentación

- [ ] `ollama serve` arrancado y modelos descargados (`qwen3.5:4b`, `gemma-4`).
- [ ] PostgreSQL y MongoDB locales activos (puertos 5432 y 27017).
- [ ] `streamlit run demo_web.py` abierto en el navegador.
- [ ] Variables de `.env` rellenadas (Groq + Neon + Atlas) si se demostrará la nube.
- [ ] Una terminal con `python src/local_llm/sql_local.py` lista por si se quiere mostrar el streaming.
- [ ] Batería de pruebas ejecutada y tablas de la sección 6 rellenadas.
- [ ] Proyector / pantalla compartida configurada.
- [ ] Tiempo ensayado: objetivo 10 min, margen 30 s.
