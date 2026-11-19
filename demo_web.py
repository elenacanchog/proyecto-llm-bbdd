import os
import json
import time
import streamlit as st
from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración de la página (DEBE ser siempre la primera instrucción de Streamlit)
# layout="wide" aprovecha todo el ancho de la pantalla para que se vea más profesional
st.set_page_config(
    page_title="Demo BBDD con LLMs",
    page_icon="🗄️",
    layout="wide"
)

# 2. Cargamos las credenciales de tu archivo .env
load_dotenv()
mongo_uri = os.getenv("MONGODB_URI")

# Esquema de la colección de Airbnb para darle contexto al LLM
ESQUEMA_AIRBNB = """
Colección: listingsAndReviews (Alojamientos turísticos de Airbnb)
Campos principales de cada documento:
- name (string): Nombre o título del alojamiento.
- property_type (string): Tipo de propiedad (ej: "Apartment", "House", "Condominium", "Loft").
- room_type (string): Tipo de habitación (ej: "Entire home/apt", "Private room").
- accommodates (int): Número máximo de huéspedes.
- bedrooms (int): Número de dormitorios.
- beds (int): Número de camas.
- price (number): Precio por noche en dólares.
- amenities (array de strings): Lista de servicios (ej: ["Wifi", "Kitchen", "Air conditioning", "Pool", "Elevator"]).
- address (objeto anidado):
  - address.street (string): Calle.
  - address.market (string): Ciudad principal (ej: "Barcelona", "New York", "Rio de Janeiro", "Istanbul").
  - address.country (string): País (ej: "Spain", "United States", "Brazil", "Turkey").
- host (objeto anidado):
  - host.host_name (string): Nombre del anfitrión.
  - host.host_is_superhost (boolean): true si es Superhost, false si no.
- review_scores (objeto anidado):
  - review_scores.review_scores_rating (int): Puntuación global de 0 a 100.
"""

# 3. Cabecera principal de la web
st.title("🗄️ Consulta de Bases de Datos mediante LLMs")
st.caption("Proyecto de Programación y Bases de Datos — Acceso en lenguaje natural (Text-to-NoSQL / Text-to-SQL)")

# 4. Menú lateral izquierdo (Sidebar)
with st.sidebar:
    st.header("⚙️ Configuración")
    
    entorno = st.radio(
        "1. Entorno del Modelo:",
        ["API en la Nube (Groq)", "Local (Ollama)"]
    )
    
    base_datos = st.selectbox(
        "2. Base de Datos:",
        ["No Relacional (MongoDB Atlas)", "Relacional (SQL - Meteoritos)"]
    )
    
    # Cambiamos la lista de modelos automáticamente según si elegimos API o Local
    if entorno == "API en la Nube (Groq)":
        modelo = st.selectbox(
            "3. Modelo en la Nube:",
            ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        )
    else:
        modelo = st.selectbox(
            "3. Modelo Local (Ollama):",
            ["llama3", "mistral"]
        )

# 5. Zona central: Caja de texto para escribir la pregunta y botón de ejecución
pregunta = st.text_input(
    "Escribe tu pregunta en español:",
    placeholder="Ej: Dime 3 apartamentos en Barcelona con Wifi y aire acondicionado, ordenados del más barato al más caro."
)

boton_consultar = st.button("🚀 Ejecutar Consulta", type="primary")

# 6. Lógica que se activa SOLO cuando pulsas el botón
if boton_consultar:
    if not pregunta.strip():
        st.warning("⚠️ Por favor, escribe una pregunta antes de pulsar el botón.")
    
    # Caso activo actualmente: API (Groq) + No Relacional (MongoDB)
    elif entorno == "API en la Nube (Groq)" and base_datos == "No Relacional (MongoDB Atlas)":
        with st.spinner(f"Traduciendo pregunta con {modelo} y consultando MongoDB Atlas..."):
            inicio_total = time.time()
            
            try:
                cliente = MongoClient(mongo_uri)
                coleccion = cliente["sample_airbnb"]["listingsAndReviews"]
                llm = ChatGroq(model=modelo, temperature=0)

                # Fase A: Generar el Pipeline de MongoDB
                prompt_generador = ChatPromptTemplate.from_messages([
                    ("system", """Eres un experto en bases de datos MongoDB. Traduce la pregunta del usuario en un Pipeline de Agregación válido de MongoDB (lista JSON de etapas como $match, $group, $sort, $project, $limit).
Usa EXCLUSIVAMENTE este esquema:
{esquema}
REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el array JSON válido. NO incluyas bloques markdown (```json), ni explicaciones.
2. Si buscas alojamientos concretos (no conteos ni medias), incluye SIEMPRE "$project" seleccionando solo campos relevantes (sin "_id") y un "$limit": 5.
3. Traduce los valores de filtro al inglés (ej: "Spain", "Wifi", "Apartment")."""),
                    ("human", "{pregunta}")
                ])

                inicio_llm = time.time()
                respuesta_llm = (prompt_generador | llm).invoke({
                    "esquema": ESQUEMA_AIRBNB,
                    "pregunta": pregunta
                })
                tiempo_query = round(time.time() - inicio_llm, 2)
                tokens = respuesta_llm.response_metadata.get("token_usage", {}).get("total_tokens", 0)

                texto_pipeline = respuesta_llm.content.strip()
                if texto_pipeline.startswith("```"):
                    texto_pipeline = texto_pipeline.replace("```json", "").replace("```", "").strip()

                # Fase B: Ejecutar en MongoDB Atlas
                pipeline = json.loads(texto_pipeline)
                registros = list(coleccion.aggregate(pipeline))
                datos_json = json_util.dumps(registros, ensure_ascii=False, indent=2)

                # Fase C: Redactar respuesta final en español
                prompt_respuesta = ChatPromptTemplate.from_messages([
                    ("system", "Responde en español de forma clara y directa basándote ÚNICAMENTE en los datos obtenidos. Si están vacíos ([]), indica que no hay resultados."),
                    ("human", "Pregunta: {pregunta}\n\nDatos de MongoDB: {datos}")
                ])

                respuesta_final = (prompt_respuesta | llm).invoke({
                    "pregunta": pregunta,
                    "datos": datos_json
                })
                tokens += respuesta_final.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                tiempo_total = round(time.time() - inicio_total, 2)

                # 7. Mostramos los resultados de forma visual y ordenada
                st.divider()
                
                # Fila superior con 3 tarjetas de métricas
                m1, m2, m3 = st.columns(3)
                m1.metric("⏱️ Tiempo generación consulta", f"{tiempo_query} s")
                m2.metric("🏁 Tiempo total respuesta", f"{tiempo_total} s")
                m3.metric("🪙 Tokens consumidos", tokens)

                st.divider()

                # Dividimos la pantalla en 2 columnas: Izquierda (Respuesta) | Derecha (Código técnico)
                col_izq, col_der = st.columns([1.2, 1])

                with col_izq:
                    st.subheader("💬 Respuesta del Asistente")
                    st.markdown(respuesta_final.content)

                with col_der:
                    st.subheader("⚙️ Consulta MongoDB Generada")
                    st.code(texto_pipeline, language="json")
                    
                    with st.expander(f"📦 Ver datos crudos devueltos por Atlas ({len(registros)} registros)"):
                        st.code(datos_json, language="json")

            except Exception as e:
                st.error(f"❌ Error durante la ejecución: {e}")
    
    else:
        st.info("🚧 Esta combinación (Base de datos Relacional o modo Local con Ollama) la conectaremos en los siguientes pasos.")