import os
import sqlite3
import time
from typing import Any
import pandas as pd
import streamlit as st
from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient
from sqlalchemy import create_engine, inspect, text
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# =============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y CREDENCIALES
# =============================================================================
st.set_page_config(
    page_title="Consulta de BBDD con LLMs",
    page_icon="🗄️",
    layout="centered"
)

load_dotenv()

CLOUD_MONGO_URI = os.getenv("MONGODB_URI")
CLOUD_POSTGRES_URI = os.getenv("POSTGRES_URI")

LOCAL_MONGO_URI = "mongodb://localhost:27017/"
LOCAL_POSTGRES_URI = "postgresql+psycopg2://postgres:12345@localhost:5432/nasa_meteoritos"

CAMPOS_PESADOS_MONGO = {
    "_id": 0,
    "reviews": 0,
    "description": 0,
    "neighborhood_overview": 0,
    "notes": 0,
    "transit": 0,
    "access": 0,
    "interaction": 0,
    "house_rules": 0,
    "images": 0,
    "host.host_about": 0,
    "host.host_url": 0,
    "host.host_thumbnail_url": 0,
    "host.host_picture_url": 0,
}

# =============================================================================
# 2. CONSTRUCTORES DE AGENTES (NOSQL Y SQL) PARA NUBE, LOCAL Y ARCHIVOS PROPIOS
# =============================================================================
def resumir_valor(valor: Any, nivel: int = 0) -> Any:
    """Compacta un documento JSON de ejemplo para mostrar el esquema al LLM."""
    if isinstance(valor, dict):
        if nivel >= 2:
            return "{...}"
        return {k: resumir_valor(v, nivel + 1) for k, v in valor.items()}
    if isinstance(valor, list):
        if not valor:
            return []
        muestra = [resumir_valor(x, nivel + 1) for x in valor[:12]]
        if len(valor) > 12:
            muestra.append(f"... (+{len(valor) - 12} elementos)")
        return muestra
    if isinstance(valor, str) and len(valor) > 40:
        return valor[:40] + "..."
    return valor


# [NUEVO] Acepta tanto una URI remota/local como una base de datos en memoria (db_personalizada)
def crear_herramientas_mongo(uri_mongo: str = None, es_local: bool = False, db_personalizada=None):
    """Crea las herramientas de MongoDB apuntando a Atlas, localhost o a un JSON en memoria."""
    if db_personalizada is not None:
        mongo_db = db_personalizada
        # En archivos propios no ocultamos campos como 'description' o 'reviews', solo '_id'
        campos_excluir = {"_id": 0}
    else:
        cliente = MongoClient(uri_mongo)
        if es_local and "sampl_airbnb" in cliente.list_database_names():
            mongo_db = cliente["sampl_airbnb"]
        else:
            mongo_db = cliente["sample_airbnb"]
        campos_excluir = CAMPOS_PESADOS_MONGO

    @tool
    def get_mongo_schema() -> str:
        """Devuelve las colecciones disponibles en MongoDB y un documento de ejemplo resumido."""
        colecciones = mongo_db.list_collection_names()
        if not colecciones:
            return "No hay colecciones en esta base de datos."
        resultado = []
        for nombre_col in colecciones:
            doc = mongo_db[nombre_col].find_one({}, campos_excluir)
            resumen = resumir_valor(doc) if doc else {}
            resultado.append(
                f"Colección '{nombre_col}' | Estructura y ejemplo de campos:\n"
                + json_util.dumps(resumen, indent=2, ensure_ascii=False)
            )
        return "\n\n".join(resultado)

    @tool
    def run_mongo(coleccion: str, operacion: str, consulta_json: str = "{}") -> str:
        """
        Ejecuta una consulta de lectura en MongoDB sobre una colección.
        - coleccion: nombre de la colección (ej. 'listingsAndReviews' o 'coleccion_json').
        - operacion: 'find', 'aggregate' o 'count'.
        - consulta_json: string JSON con el filtro (en find/count) o lista JSON de etapas (en aggregate).
        """
        try:
            col = mongo_db[coleccion]
            consulta = json_util.loads(consulta_json) if isinstance(consulta_json, str) else consulta_json

            if operacion == "find":
                filas = list(col.find(consulta, campos_excluir).limit(12))
                return json_util.dumps(filas, indent=2, ensure_ascii=False) if filas else "No se encontraron documentos."

            elif operacion == "aggregate":
                if isinstance(consulta, dict):
                    consulta = [{k: v} for k, v in consulta.items()]
                tiene_proyeccion_o_grupo = any(
                    any(k in etapa for k in ("$project", "$group", "$count"))
                    for etapa in consulta if isinstance(etapa, dict)
                )
                pipeline = consulta if tiene_proyeccion_o_grupo else consulta + [{"$project": campos_excluir}]
                filas = list(col.aggregate(pipeline))
                return json_util.dumps(filas, indent=2, ensure_ascii=False) if filas else "La agregación no devolvió resultados."

            elif operacion == "count":
                return f"Total de documentos: {col.count_documents(consulta)}"

            return "Operación no válida. Usa únicamente: 'find', 'aggregate' o 'count'."
        except Exception as e:
            return f"Error MongoDB: {e}"

    return [get_mongo_schema, run_mongo]


# [NUEVO] Acepta tanto una URI de PostgreSQL como un motor SQLite en memoria (engine_personalizado)
def crear_herramientas_sql(uri_postgres: str = None, engine_personalizado=None):
    """Crea las herramientas SQL apuntando a Neon, localhost o a un CSV cargado en SQLite."""
    engine = engine_personalizado if engine_personalizado is not None else create_engine(uri_postgres)

    @tool
    def get_schema() -> str:
        """Devuelve las tablas disponibles y las columnas con sus tipos de datos en la base de datos SQL."""
        inspector = inspect(engine)
        resultado = []
        for tabla in inspector.get_table_names():
            columnas = [f"{col['name']} ({col['type']})" for col in inspector.get_columns(tabla)]
            resultado.append(f"Tabla '{tabla}': {', '.join(columnas)}")
        return "\n".join(resultado)

    @tool
    def run_sql(consulta: str) -> str:
        """Ejecuta cualquier consulta SQL (SELECT, INSERT, UPDATE, DELETE) en la base de datos."""
        try:
            with engine.begin() as conn:
                resultado = conn.execute(text(consulta))
                if resultado.returns_rows:
                    filas = resultado.fetchall()
                    return str(filas) if filas else "La consulta no devolvió resultados."
                return f"Operación ejecutada correctamente. Filas afectadas: {resultado.rowcount}"
        except Exception as e:
            return f"Error SQL: {e}"

    return [get_schema, run_sql]


def obtener_llm(entorno: str, modelo: str):
    """Instancia el modelo LLM en la nube (Groq) o en local (Ollama)."""
    if entorno == "API en la Nube (Groq)":
        return ChatGroq(model=modelo, temperature=0, max_tokens=800)
    else:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "No está instalada la librería 'langchain-ollama' en este equipo. "
                "El modo Local debe ejecutarse desde el ordenador que tenga Ollama configurado."
            )
        return ChatOllama(model=modelo, temperature=0, num_ctx=8192)


# =============================================================================
# 3. INTERFAZ VISUAL PASO A PASO (SIN BARRA LATERAL)
# =============================================================================
st.title("🗄️ Consulta de Bases de Datos con LLMs")
st.markdown("Haz preguntas en lenguaje natural y deja que un Agente IA consulte la base de datos por ti.")
st.divider()

st.markdown("**1. ¿Cómo quieres ejecutar el modelo?**")
entorno = st.pills(
    "Entorno de ejecución",
    ["API en la Nube (Groq)", "Local (Ollama)"],
    default="API en la Nube (Groq)",
    label_visibility="collapsed"
)
if not entorno:
    entorno = "API en la Nube (Groq)"

st.markdown("**2. ¿Qué base de datos quieres consultar?**")
# [NUEVO] Añadida la tercera opción para subir archivos propios
base_datos = st.pills(
    "Base de datos",
    [
        "No Relacional (MongoDB - Airbnb)",
        "Relacional (PostgreSQL - Meteoritos)",
        "Archivo propio (.CSV / .JSON)"
    ],
    default="No Relacional (MongoDB - Airbnb)",
    label_visibility="collapsed"
)
if not base_datos:
    base_datos = "No Relacional (MongoDB - Airbnb)"

# [NUEVO] Si se elige archivo propio, mostramos el selector de archivos
archivo_subido = None
if base_datos == "Archivo propio (.CSV / .JSON)":
    archivo_subido = st.file_uploader(
        "Sube tu archivo de datos (.csv para consulta SQL o .json para consulta NoSQL):",
        type=["csv", "json"]
    )

st.markdown("**3. ¿Qué modelo quieres usar?**")
if entorno == "API en la Nube (Groq)":
    opciones_modelos = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
else:
    opciones_modelos = ["qwen3.5:4b", "gemma-4:latest"]

modelo = st.pills(
    "Modelo LLM",
    opciones_modelos,
    default=opciones_modelos[0],
    key=f"pills_modelo_{entorno}",
    label_visibility="collapsed"
)
if not modelo:
    modelo = opciones_modelos[0]

st.divider()

if "No Relacional" in base_datos:
    ejemplo_placeholder = "Ej: Dime 3 apartamentos en Barcelona que tengan Wifi y aire acondicionado, ordenados del más barato al más caro."
elif "Relacional" in base_datos:
    ejemplo_placeholder = "Ej: ¿Cuáles son los 3 meteoritos más pesados que cayeron después del año 1900?"
else:
    ejemplo_placeholder = "Ej: ¿Cuántos registros hay en total y cuáles son los 3 valores más altos?"

pregunta = st.text_area(
    "**4. Escribe tu consulta en español:**",
    placeholder=ejemplo_placeholder,
    height=100
)

boton_consultar = st.button("🚀 Ejecutar Consulta", type="primary", use_container_width=True)

# =============================================================================
# 4. EJECUCIÓN DEL AGENTE Y VISUALIZACIÓN DE RESULTADOS
# =============================================================================
if boton_consultar:
    if not pregunta.strip():
        st.warning("⚠️ Por favor, escribe una pregunta antes de pulsar el botón.")
    elif base_datos == "Archivo propio (.CSV / .JSON)" and archivo_subido is None:
        st.warning("⚠️ Por favor, sube primero un archivo .csv o .json antes de ejecutar la consulta.")
    else:
        es_local = (entorno == "Local (Ollama)")

        with st.spinner(f"Agente pensando y consultando los datos con {modelo}..."):
            inicio_total = time.time()
            try:
                llm = obtener_llm(entorno, modelo)

                # --- CASO A: ARCHIVO PROPIO SUBIDO POR EL USUARIO (.CSV o .JSON) ---
                if base_datos == "Archivo propio (.CSV / .JSON)":
                    nombre_archivo = archivo_subido.name.lower()

                    if nombre_archivo.endswith(".csv"):
                        # Leemos el CSV detectando automáticamente el separador (, o ;)
                        df = pd.read_csv(archivo_subido, sep=None, engine="python")
                        # Limpiamos espacios y paréntesis en columnas para facilitar el SQL al agente
                        df.columns = [
                            c.strip().replace(" ", "_").replace("(", "").replace(")", "")
                            for c in df.columns
                        ]
                        # Creamos una BD SQLite temporal en memoria RAM
                        conn_sqlite = sqlite3.connect(":memory:", check_same_thread=False)
                        engine_sqlite = create_engine("sqlite://", creator=lambda: conn_sqlite)
                        df.to_sql("datos_csv", engine_sqlite, index=False, if_exists="replace")

                        herramientas = crear_herramientas_sql(engine_personalizado=engine_sqlite)
                        nombre_tool_ejecucion = "run_sql"
                        lenguaje_codigo = "sql"
                        es_mongo = False
                        prompt_sistema = """Eres un agente experto en bases de datos SQL (SQLite).
Dada una petición del usuario, construye la consulta SQL sintácticamente correcta, ejecútala y responde en español usando los resultados obtenidos.
Reglas obligatorias:
1. Revisa SIEMPRE primero qué tabla y columnas existen con get_schema antes de escribir SQL.
2. En consultas SELECT, limita los resultados a un máximo de 5 filas salvo que el usuario pida otra cantidad.
3. Responde siempre en español basándote únicamente en los datos devueltos."""

                    elif nombre_archivo.endswith(".json"):
                        try:
                            import mongomock
                        except ImportError:
                            raise ImportError("Falta instalar 'mongomock'. Ejecuta: pip install mongomock")

                        contenido_texto = archivo_subido.getvalue().decode("utf-8")
                        datos_cargados = json_util.loads(contenido_texto)
                        if isinstance(datos_cargados, dict):
                            datos_cargados = [datos_cargados]

                        # Creamos un servidor MongoDB simulado en memoria RAM
                        cliente_mock = mongomock.MongoClient()
                        db_mock = cliente_mock["bd_usuario"]
                        if datos_cargados:
                            db_mock["coleccion_json"].insert_many(datos_cargados)

                        herramientas = crear_herramientas_mongo(db_personalizada=db_mock)
                        nombre_tool_ejecucion = "run_mongo"
                        lenguaje_codigo = "json"
                        es_mongo = True
                        prompt_sistema = """Eres un agente experto en bases de datos MongoDB.
Responde siempre en español de forma clara y directa usando exclusivamente los datos obtenidos de la colección.
Reglas obligatorias:
1. Llama SIEMPRE primero a get_mongo_schema para ver el nombre de la colección y los campos reales del JSON.
2. Para filtrar, ordenar, limitar resultados o calcular medias, usa operacion='aggregate' pasando en consulta_json una LISTA JSON de etapas.
3. Si una consulta no devuelve resultados, revisa los nombres exactos y tipos de datos en el esquema antes de reintentar."""

                # --- CASO B: MONGODB ATLAS (NUBE) O LOCALHOST ---
                elif "No Relacional" in base_datos:
                    es_mongo = True
                    uri = LOCAL_MONGO_URI if es_local else CLOUD_MONGO_URI
                    if not uri:
                        raise ValueError("Falta la variable MONGODB_URI en el archivo .env")
                    herramientas = crear_herramientas_mongo(uri_mongo=uri, es_local=es_local)
                    nombre_tool_ejecucion = "run_mongo"
                    lenguaje_codigo = "json"
                    prompt_sistema = """Eres un agente experto en bases de datos MongoDB.
Responde siempre en español de forma clara y directa usando exclusivamente los datos obtenidos de la base de datos.
Reglas obligatorias:
1. Llama SIEMPRE primero a get_mongo_schema para ver qué colecciones existen y los nombres exactos de sus campos.
2. Traduce los valores de filtro del usuario al inglés (ej. "Spain", "Apartment", "Wifi", "Air conditioning").
3. Para filtrar, ordenar, limitar resultados o calcular medias, usa operacion='aggregate' pasando en consulta_json una LISTA JSON de etapas.
4. Si incluyes "$project", proyecta siempre los campos usados en "$match" y subcampos concretos (ej. "host.host_name": 1)."""

                # --- CASO C: POSTGRESQL NEON (NUBE) O LOCALHOST ---
                else:
                    es_mongo = False
                    uri = LOCAL_POSTGRES_URI if es_local else CLOUD_POSTGRES_URI
                    if not uri:
                        raise ValueError("Falta la variable POSTGRES_URI en el archivo .env")
                    herramientas = crear_herramientas_sql(uri_postgres=uri)
                    nombre_tool_ejecucion = "run_sql"
                    lenguaje_codigo = "sql"
                    prompt_sistema = """Eres un agente experto en bases de datos PostgreSQL.
Dada una petición del usuario, construye la consulta SQL sintácticamente correcta, ejecútala y responde en español usando los resultados obtenidos.
Reglas obligatorias:
1. Revisa SIEMPRE primero qué tablas y columnas existen con get_schema antes de escribir SQL.
2. En consultas SELECT, limita los resultados a un máximo de 5 filas salvo que el usuario pida otra cantidad.
3. Si ordenas de forma descendente (DESC), añade NULLS LAST para evitar que los valores nulos salgan primero.
4. Puedes ejecutar SELECT, INSERT, UPDATE y DELETE cuando el usuario lo solicite."""

                # --- EJECUCIÓN COMÚN DEL AGENTE ---
                agente = create_agent(
                    model=llm,
                    tools=herramientas,
                    system_prompt=prompt_sistema
                )

                resultado_agente = agente.invoke(
                    {"messages": [("user", pregunta)]},
                    config={"recursion_limit": 10}
                )
                mensajes = resultado_agente["messages"]

                consulta_ejecutada = "No se ejecutó ninguna consulta en la base de datos."
                datos_crudos = "Sin datos"
                intentos = 0
                tokens_totales = 0

                for msg in mensajes:
                    if getattr(msg, "usage_metadata", None):
                        tokens_totales += msg.usage_metadata.get("total_tokens", 0)
                    elif hasattr(msg, "response_metadata") and msg.response_metadata:
                        uso = msg.response_metadata.get("token_usage", {})
                        tokens_totales += uso.get("total_tokens", 0)

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for llamada in msg.tool_calls:
                            if llamada["name"] == nombre_tool_ejecucion:
                                intentos += 1
                                if es_mongo:
                                    op = llamada["args"].get("operacion", "")
                                    q = llamada["args"].get("consulta_json", "")
                                    consulta_ejecutada = f"// Operación: {op}\n{q}"
                                else:
                                    consulta_ejecutada = llamada["args"].get("consulta", "")

                    if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == nombre_tool_ejecucion:
                        datos_crudos = msg.content

                respuesta_final = mensajes[-1].content
                tiempo_total = round(time.time() - inicio_total, 2)

                # --- MOSTRAR RESULTADOS EN PANTALLA ---
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("🔄 Consultas a BD", intentos)
                m2.metric("⏱️ Tiempo total", f"{tiempo_total} s")
                m3.metric("🪙 Tokens consumidos", tokens_totales)

                st.divider()
                col_izq, col_der = st.columns([1.2, 1])

                with col_izq:
                    st.subheader("💬 Respuesta del Asistente")
                    st.markdown(respuesta_final)

                with col_der:
                    st.subheader("⚙️ Consulta Generada")
                    st.code(consulta_ejecutada, language=lenguaje_codigo)

                    with st.expander("📦 Ver datos crudos devueltos por la BD"):
                        st.code(datos_crudos, language=lenguaje_codigo)

            except Exception as e:
                st.error(f"❌ Error durante la ejecución: {e}")