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
    page_title="DataChat · Consulta de BBDD con LLMs",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

CLOUD_MONGO_URI = os.getenv("MONGODB_URI")
CLOUD_POSTGRES_URI = os.getenv("POSTGRES_URI")

LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI")
LOCAL_MONGO_DB = os.getenv("LOCAL_MONGO_DB")
LOCAL_POSTGRES_URI = os.getenv("LOCAL_POSTGRES_URI")

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

# Constantes de interfaz
OPT_NUBE = "API en la Nube (Groq)"
OPT_LOCAL = "Local (Ollama)"
OPT_MONGO = "No Relacional (MongoDB - Airbnb)"
OPT_PG = "Relacional (PostgreSQL - Meteoritos)"
OPT_ARCHIVO = "Archivo propio (.CSV / .JSON)"

# Enlace al repositorio
REPO_GITHUB_URL = "https://github.com/elenacanchog/proyecto-llm-bbdd"

EJEMPLOS = {
    OPT_MONGO: [
        "Dime 3 apartamentos en Barcelona con Wifi y aire acondicionado, del más barato al más caro.",
        "¿Cuál es el precio medio por noche de los apartamentos en España?",
        "¿Cuántos alojamientos hay en total en la colección?",
    ],
    OPT_PG: [
        "¿Cuáles son los 3 meteoritos más pesados que cayeron después del año 1900?",
        "¿Cuántos meteoritos hay registrados en total?",
        "¿Qué año tiene más caídas de meteoritos registradas?",
    ],
    OPT_ARCHIVO: [
        "¿Cuántos registros hay en total y cuáles son los 3 valores más altos?",
        "Dame un resumen de las columnas que contiene el archivo.",
        "Muéstrame los 5 primeros registros.",
    ],
}

# =============================================================================
# 2. CONSTRUCTORES DE AGENTES (NOSQL Y SQL) PARA NUBE, LOCAL Y ARCHIVOS PROPIOS
#    (lógica sin cambios respecto a la versión anterior)
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


def crear_herramientas_mongo(uri_mongo: str = None, es_local: bool = False, db_personalizada=None):
    """Crea las herramientas de MongoDB apuntando a Atlas, localhost o a un JSON en memoria."""
    if db_personalizada is not None:
        mongo_db = db_personalizada
        # En archivos propios no ocultamos campos como 'description' o 'reviews', solo '_id'
        campos_excluir = {"_id": 0}
    else:
        cliente = MongoClient(uri_mongo)
        if es_local and LOCAL_MONGO_DB:
            mongo_db = cliente[LOCAL_MONGO_DB]
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
    if entorno == OPT_NUBE:
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
# 3. ESTILOS (CSS) Y COMPONENTES VISUALES
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --accent-1: #6366f1;
    --accent-2: #8b5cf6;
    --accent-3: #06b6d4;
    --ok: #10b981;
    --warn: #f59e0b;
    --err: #ef4444;
    --card-bg: rgba(127, 127, 127, 0.06);
    --card-border: rgba(127, 127, 127, 0.22);
}

html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ---- Layout centrado y con ancho máximo ---- */
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    margin: 0 auto;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---- Hero ---- */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 2.6rem 2.8rem;
    margin-bottom: 1.6rem;
    color: #fff;
    background:
        radial-gradient(circle at 85% 15%, rgba(6, 182, 212, 0.55), transparent 45%),
        radial-gradient(circle at 10% 90%, rgba(236, 72, 153, 0.35), transparent 45%),
        linear-gradient(135deg, #4338ca 0%, #6d28d9 55%, #7c3aed 100%);
    box-shadow: 0 20px 45px -15px rgba(99, 102, 241, 0.55);
}
.hero h1 {
    margin: 0 0 .4rem 0;
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #fff;
    line-height: 1.15;
}
.hero p {
    margin: 0;
    max-width: 640px;
    font-size: 1.05rem;
    line-height: 1.55;
    color: rgba(255, 255, 255, 0.88);
}
.hero-badges { margin-top: 1.3rem; display: flex; flex-wrap: wrap; gap: .5rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .35rem .8rem;
    font-size: .8rem; font-weight: 600;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    backdrop-filter: blur(6px);
}

/* ---- Tarjetas (contenedores con borde) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    box-shadow: 0 6px 24px -12px rgba(0, 0, 0, 0.25);
}

/* ---- Títulos de paso ---- */
.step {
    display: flex; align-items: center; gap: .7rem;
    margin: .2rem 0 .55rem 0;
}
.step-num {
    flex: 0 0 auto;
    width: 28px; height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 50%;
    font-size: .85rem; font-weight: 700; color: #fff;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    box-shadow: 0 4px 10px -3px rgba(99, 102, 241, .7);
}
.step-title { font-size: 1rem; font-weight: 700; letter-spacing: -0.01em; }
.step-hint { font-size: .82rem; opacity: .65; margin: -.25rem 0 .6rem 2.35rem; }

.section-title {
    font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em;
    margin: .2rem 0 .8rem 0;
}

/* ---- Pills de Streamlit ---- */
button[data-testid="stBaseButton-pills"] {
    border-radius: 999px;
    font-weight: 500;
    border: 1px solid var(--card-border);
    transition: all .15s ease;
}
button[data-testid="stBaseButton-pills"]:hover {
    border-color: var(--accent-1);
    transform: translateY(-1px);
}
button[data-testid="stBaseButton-pillsActive"] {
    border-radius: 999px;
    font-weight: 600;
    color: #fff !important;
    border: 1px solid transparent;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    box-shadow: 0 6px 14px -6px rgba(99, 102, 241, .8);
}
button[data-testid="stBaseButton-pillsActive"] * { color: #fff !important; }

/* ---- Botón primario ---- */
button[data-testid="stBaseButton-primary"], button[kind="primary"] {
    border: none;
    border-radius: 14px;
    padding: .7rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    color: #fff;
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 60%, #a855f7 100%);
    box-shadow: 0 10px 22px -10px rgba(99, 102, 241, .9);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
    transform: translateY(-2px);
    filter: brightness(1.07);
    box-shadow: 0 14px 26px -10px rgba(99, 102, 241, 1);
    color: #fff;
}

/* ---- Botones secundarios (ejemplos) ---- */
button[data-testid="stBaseButton-secondary"] {
    border-radius: 12px;
    border: 1px dashed var(--card-border);
    font-size: .82rem;
    text-align: left;
    transition: all .15s ease;
}
button[data-testid="stBaseButton-secondary"]:hover {
    border-style: solid;
    border-color: var(--accent-1);
    color: var(--accent-1);
}

/* ---- Text area / uploader ---- */
.stTextArea textarea {
    border-radius: 14px;
    font-size: 1rem;
    line-height: 1.5;
    padding: .9rem 1rem;
}
.stTextArea textarea:focus {
    border-color: var(--accent-1);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, .25);
}
div[data-testid="stFileUploader"] section {
    border-radius: 14px;
    border: 1.5px dashed var(--accent-1);
    background: rgba(99, 102, 241, .05);
}

/* ---- Chips de estado ---- */
.chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .4rem; }
.chip {
    display: inline-flex; align-items: center; gap: .35rem;
    padding: .25rem .7rem;
    font-size: .76rem; font-weight: 600;
    border-radius: 999px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
}
.chip.ok   { color: var(--ok);   border-color: rgba(16,185,129,.45); background: rgba(16,185,129,.10); }
.chip.warn { color: var(--warn); border-color: rgba(245,158,11,.45); background: rgba(245,158,11,.10); }
.chip.info { color: var(--accent-1); border-color: rgba(99,102,241,.45); background: rgba(99,102,241,.10); }

/* ---- Tarjetas de métricas ---- */
.metric-card {
    display: flex; align-items: center; gap: 1rem;
    padding: 1.1rem 1.3rem;
    border-radius: 18px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    box-shadow: 0 6px 24px -14px rgba(0, 0, 0, 0.3);
}
.metric-icon {
    width: 46px; height: 46px; flex: 0 0 auto;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(99,102,241,.22), rgba(6,182,212,.22));
}
.metric-label { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; opacity: .65; }
.metric-value { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.2; }

/* ---- Respuesta ---- */
.answer-box {
    padding: 1.3rem 1.5rem;
    border-radius: 16px;
    border-left: 4px solid var(--accent-1);
    background: linear-gradient(90deg, rgba(99,102,241,.10), rgba(99,102,241,.02));
}

/* ---- Código ---- */
code, pre, .stCode { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
div[data-testid="stCode"] { border-radius: 14px; }

/* ---- Pestañas ---- */
button[data-baseweb="tab"] { font-weight: 600; }

/* ---- Pie ---- */
.footer {
    margin-top: 2.2rem; text-align: center;
    font-size: .8rem; opacity: .55;
}

@media (max-width: 768px) {
    .hero { padding: 1.6rem 1.3rem; }
    .hero h1 { font-size: 1.7rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def paso(numero: int, titulo: str, ayuda: str = "") -> None:
    st.markdown(
        f'<div class="step"><span class="step-num">{numero}</span>'
        f'<span class="step-title">{titulo}</span></div>',
        unsafe_allow_html=True,
    )
    if ayuda:
        st.markdown(f'<div class="step-hint">{ayuda}</div>', unsafe_allow_html=True)


def chip(texto: str, tipo: str = "") -> str:
    return f'<span class="chip {tipo}">{texto}</span>'


def tarjeta_metrica(icono: str, etiqueta: str, valor: Any) -> str:
    return (
        f'<div class="metric-card"><div class="metric-icon">{icono}</div>'
        f'<div><div class="metric-label">{etiqueta}</div>'
        f'<div class="metric-value">{valor}</div></div></div>'
    )


def callback_ejemplo(texto: str) -> None:
    st.session_state["pregunta"] = texto


# Estado de sesión
if "pregunta" not in st.session_state:
    st.session_state["pregunta"] = ""
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# =============================================================================
# 4. INTERFAZ VISUAL
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🗄️ DataChat · Consulta de Bases de Datos con IA</h1>
        <p>Haz preguntas en lenguaje natural y deja que un agente inteligente consulte
        la base de datos por ti: SQL, MongoDB o tus propios archivos.</p>
        <div class="hero-badges">
            <span class="hero-badge">☁️ API en la nube o 🖥️ Local</span>
            <span class="hero-badge">🍃 MongoDB</span>
            <span class="hero-badge">🐘 PostgreSQL</span>
            <span class="hero-badge">📄 CSV / JSON propios</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_config, col_consulta = st.columns([1, 1.45], gap="large")

# ------------------------- COLUMNA IZQUIERDA: CONFIGURACIÓN -------------------------
with col_config:
    with st.container(border=True):
        st.markdown('<div class="section-title">⚙️ Configuración</div>', unsafe_allow_html=True)

        paso(1, "Entorno de ejecución", "¿Dónde se ejecuta el modelo de lenguaje?")
        entorno = st.pills(
            "Entorno de ejecución",
            [OPT_NUBE, OPT_LOCAL],
            default=OPT_NUBE,
            label_visibility="collapsed",
            key="pills_entorno",
        )
        if not entorno:
            entorno = OPT_NUBE
        
        if entorno == OPT_LOCAL:
            st.info(
                "🖥️ **Modo Local:** Solo funciona si ejecutas `demo_web.py` en tu propio ordenador "
                "con Ollama, MongoDB y PostgreSQL activos. En la versión web pública usa **API en la Nube (Groq)**.\n\n"
                f"📖 [Ver guía de instalación y requisitos en GitHub]({REPO_GITHUB_URL})"
            )

        st.write("")
        paso(2, "Fuente de datos", "Elige una base de datos o sube tu propio archivo.")
        base_datos = st.pills(
            "Base de datos",
            [OPT_MONGO, OPT_PG, OPT_ARCHIVO],
            default=OPT_MONGO,
            label_visibility="collapsed",
            key="pills_bd",
        )
        if not base_datos:
            base_datos = OPT_MONGO

        archivo_subido = None
        if base_datos == OPT_ARCHIVO:
            archivo_subido = st.file_uploader(
                "Sube tu archivo de datos (.csv para consulta SQL o .json para consulta NoSQL):",
                type=["csv", "json"],
            )
            if archivo_subido is not None:
                tam_kb = archivo_subido.size / 1024
                tipo_motor = "SQL (SQLite)" if archivo_subido.name.lower().endswith(".csv") else "NoSQL (MongoDB simulado)"
                st.markdown(
                    '<div class="chips">'
                    + chip(f"📎 {archivo_subido.name}", "info")
                    + chip(f"{tam_kb:,.1f} KB")
                    + chip(tipo_motor, "ok")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if archivo_subido.name.lower().endswith(".csv"):
                    with st.expander("👁️ Vista previa del archivo"):
                        try:
                            vista = pd.read_csv(archivo_subido, sep=None, engine="python", nrows=5)
                            st.dataframe(vista, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.caption(f"No se pudo generar la vista previa: {e}")
                        finally:
                            archivo_subido.seek(0)

        st.write("")
        paso(3, "Modelo LLM", "Los modelos disponibles dependen del entorno elegido.")
        if entorno == OPT_NUBE:
            opciones_modelos = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        else:
            opciones_modelos = ["qwen3.5:4b", "gemma-4:latest"]

        modelo = st.pills(
            "Modelo LLM",
            opciones_modelos,
            default=opciones_modelos[0],
            key=f"pills_modelo_{entorno}",
            label_visibility="collapsed",
        )
        if not modelo:
            modelo = opciones_modelos[0]

        # Estado de las credenciales / conexión
        st.write("")
        chips_estado = [chip(f"🤖 {modelo}", "info")]
        if entorno == OPT_NUBE:
            chips_estado.append(
                chip("Groq API key ✓", "ok") if os.getenv("GROQ_API_KEY") else chip("Falta GROQ_API_KEY", "warn")
            )
        else:
            chips_estado.append(chip("Requiere Ollama local", "warn"))
        if base_datos == OPT_MONGO and entorno == OPT_NUBE:
            chips_estado.append(chip("MongoDB URI ✓", "ok") if CLOUD_MONGO_URI else chip("Falta MONGODB_URI", "warn"))
        elif base_datos == OPT_PG and entorno == OPT_NUBE:
            chips_estado.append(chip("PostgreSQL URI ✓", "ok") if CLOUD_POSTGRES_URI else chip("Falta POSTGRES_URI", "warn"))
        st.markdown('<div class="chips">' + "".join(chips_estado) + "</div>", unsafe_allow_html=True)

# ------------------------- COLUMNA DERECHA: CONSULTA -------------------------
if base_datos == OPT_MONGO:
    ejemplo_placeholder = "Ej: Dime 3 apartamentos en Barcelona que tengan Wifi y aire acondicionado, ordenados del más barato al más caro."
elif base_datos == OPT_PG:
    ejemplo_placeholder = "Ej: ¿Cuáles son los 3 meteoritos más pesados que cayeron después del año 1900?"
else:
    ejemplo_placeholder = "Ej: ¿Cuántos registros hay en total y cuáles son los 3 valores más altos?"

with col_consulta:
    with st.container(border=True):
        st.markdown('<div class="section-title">💬 Tu consulta</div>', unsafe_allow_html=True)
        paso(4, "Escribe tu pregunta en español", "Describe lo que quieres saber; el agente generará la consulta por ti.")

        pregunta = st.text_area(
            "Consulta en español",
            key="pregunta",
            placeholder=ejemplo_placeholder,
            height=140,
            label_visibility="collapsed",
        )

        st.markdown("<div class='step-hint' style='margin-left:0'>💡 Prueba con un ejemplo:</div>", unsafe_allow_html=True)
        for i, ej in enumerate(EJEMPLOS[base_datos]):
            st.button(
                ej,
                key=f"ej_{base_datos}_{i}",
                on_click=callback_ejemplo,
                args=(ej,),
                use_container_width=True,
            )

        st.write("")
        boton_consultar = st.button("🚀 Ejecutar Consulta", type="primary", use_container_width=True)

# =============================================================================
# 5. EJECUCIÓN DEL AGENTE
# =============================================================================
if boton_consultar:
    if not pregunta.strip():
        st.warning("⚠️ Por favor, escribe una pregunta antes de pulsar el botón.")
    elif base_datos == OPT_ARCHIVO and archivo_subido is None:
        st.warning("⚠️ Por favor, sube primero un archivo .csv o .json antes de ejecutar la consulta.")
    else:
        es_local = (entorno == OPT_LOCAL)

        with st.spinner(f"Agente pensando y consultando los datos con {modelo}..."):
            inicio_total = time.time()
            try:
                llm = obtener_llm(entorno, modelo)

                # --- CASO A: ARCHIVO PROPIO SUBIDO POR EL USUARIO (.CSV o .JSON) ---
                if base_datos == OPT_ARCHIVO:
                    nombre_archivo = archivo_subido.name.lower()

                    if nombre_archivo.endswith(".csv"):
                        archivo_subido.seek(0)
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
                elif base_datos == OPT_MONGO:
                    es_mongo = True
                    uri = LOCAL_MONGO_URI if es_local else CLOUD_MONGO_URI
                    if not uri:
                        var_faltante = "LOCAL_MONGO_URI" if es_local else "MONGODB_URI"
                        raise ValueError(f"Falta la variable {var_faltante} en el archivo .env")
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
                        var_faltante = "LOCAL_POSTGRES_URI" if es_local else "POSTGRES_URI"
                        raise ValueError(f"Falta la variable {var_faltante} en el archivo .env")
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

                st.session_state["resultado"] = {
                    "ok": True,
                    "pregunta": pregunta,
                    "entorno": entorno,
                    "base_datos": base_datos,
                    "modelo": modelo,
                    "intentos": intentos,
                    "tiempo": tiempo_total,
                    "tokens": tokens_totales,
                    "respuesta": respuesta_final,
                    "consulta": consulta_ejecutada,
                    "datos": datos_crudos,
                    "lenguaje": lenguaje_codigo,
                }
                st.session_state["historial"].insert(0, {
                    "pregunta": pregunta,
                    "modelo": modelo,
                    "base_datos": base_datos,
                    "tiempo": tiempo_total,
                    "respuesta": respuesta_final,
                })
                st.session_state["historial"] = st.session_state["historial"][:10]

            except Exception as e:
                msg_error = str(e)
                if "429" in msg_error or "rate_limit" in msg_error.lower():
                    msg_error = (
                        "⏳ Se ha alcanzado el límite gratuito de consultas por minuto de Groq. "
                        "Espera unos 30 segundos o prueba cambiando a otro modelo (por ejemplo, gpt-oss-20b)."
                    )
                elif "Connection refused" in msg_error or "No está instalada la librería 'langchain-ollama'" in msg_error:
                    msg_error = (
                        "🔌 No se pudo conectar con el entorno local (Ollama / bases de datos en localhost). "
                        "Si estás usando la web desplegada, selecciona 'API en la Nube (Groq)' en el Paso 1."
                    )
                st.session_state["resultado"] = {"ok": False, "error": msg_error}

# =============================================================================
# 6. VISUALIZACIÓN DE RESULTADOS
# =============================================================================
res = st.session_state["resultado"]

if res is not None:
    st.write("")
    st.markdown('<div class="section-title">📊 Resultados</div>', unsafe_allow_html=True)

    if not res["ok"]:
        st.error(f"❌ Error durante la ejecución: {res['error']}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.markdown(tarjeta_metrica("🔄", "Consultas a BD", res["intentos"]), unsafe_allow_html=True)
        m2.markdown(tarjeta_metrica("⏱️", "Tiempo total", f"{res['tiempo']} s"), unsafe_allow_html=True)
        m3.markdown(tarjeta_metrica("🪙", "Tokens consumidos", f"{res['tokens']:,}"), unsafe_allow_html=True)

        st.write("")
        col_izq, col_der = st.columns([1.2, 1], gap="large")

        with col_izq:
            with st.container(border=True):
                st.subheader("💬 Respuesta del Asistente")
                st.caption(f"❓ {res['pregunta']}")
                st.markdown(res["respuesta"])

        with col_der:
            with st.container(border=True):
                st.subheader("⚙️ Consulta Generada")
                st.code(res["consulta"], language=res["lenguaje"])

                with st.expander("📦 Ver datos crudos devueltos por la BD"):
                    st.code(res["datos"], language=res["lenguaje"])

# Historial de la sesión
if st.session_state["historial"]:
    st.write("")
    with st.expander(f"🕘 Historial de consultas de esta sesión ({len(st.session_state['historial'])})"):
        for item in st.session_state["historial"]:
            st.markdown(f"**{item['pregunta']}**")
            st.caption(f"{item['base_datos']} · {item['modelo']} · {item['tiempo']} s")
            st.markdown(item["respuesta"])
            st.divider()

st.markdown(
    f'<div class="footer">DataChat · Agentes LLM sobre MongoDB, PostgreSQL y archivos propios · '
    f'<a href="{REPO_GITHUB_URL}" target="_blank" style="color: inherit; text-decoration: underline;">'
    f'Código fuente y guía en GitHub</a></div>',
    unsafe_allow_html=True,
)