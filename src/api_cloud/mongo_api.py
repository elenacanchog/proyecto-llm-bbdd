import os
import json
import time
import pandas as pd
from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# 1. Cargamos las variables del archivo .env
load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
groq_api_key = os.getenv("GROQ_API_KEY")

# PARÁMETRO DEL MODELO: Cambia aquí el nombre por cualquiera disponible en tu cuenta de Groq
MODELO_POR_DEFECTO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# 2. Definimos el "mapa" (esquema) de sample_airbnb para que el LLM sepa qué campos existen
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

def probar_conexion_airbnb():
    print("Conectando a MongoDB Atlas (sample_airbnb)...")
    
    if not mongo_uri:
        print("ERROR: No se encontró la variable MONGODB_URI en el archivo .env")
        return

    # 2. Conectamos con el cluster
    cliente = MongoClient(mongo_uri)
    
    # 3. Seleccionamos la base de datos 'sample_airbnb' y la colección 'listingsAndReviews'
    db = cliente["sample_airbnb"]
    coleccion = db["listingsAndReviews"]
    
    # 4. Contamos los alojamientos y buscamos uno en España como ejemplo
    total = coleccion.count_documents({})
    ejemplo = coleccion.find_one(
        {"address.country": "Spain"},
        {
            "name": 1,
            "property_type": 1,
            "price": 1,
            "address.market": 1,
            "host.host_name": 1,
            "amenities": 1,
            "_id": 0
        }
    )
    
    print(f"¡Conexión exitosa! Total de alojamientos disponibles: {total}")
    print("\nEjemplo de documento JSON anidado recuperado:")
    print(ejemplo)

def consultar_airbnb_con_ia(pregunta_usuario: str, modelo: str = MODELO_POR_DEFECTO):
    print(f"\nModelo utilizado: {modelo}")
    print(f"Pregunta: '{pregunta_usuario}'")
    print("-" * 60)

    # 3. Conectamos a MongoDB Atlas y a Groq usando el parámetro 'modelo'
    cliente = MongoClient(mongo_uri)
    coleccion = cliente["sample_airbnb"]["listingsAndReviews"]
    llm = ChatGroq(model=modelo, temperature=0)

    # 4. Prompt 1: Traducir pregunta a Pipeline de MongoDB
    prompt_generador = ChatPromptTemplate.from_messages([
        ("system", """Eres un experto en bases de datos MongoDB. Tu única tarea es traducir la pregunta del usuario en un Pipeline de Agregación válido de MongoDB (una lista JSON de etapas como $match, $group, $sort, $project, $limit).

Usa EXCLUSIVAMENTE este esquema:
{esquema}

REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el array JSON válido. NO incluyas bloques de código markdown (```json), ni explicaciones, ni texto adicional.
2. Si buscas alojamientos concretos (no conteos ni medias), incluye SIEMPRE una etapa "$project" seleccionando solo los campos relevantes (y excluyendo "_id": 0) y un "$limit": 5 para no devolver documentos gigantes.
3. Los valores de texto en la base de datos están en inglés (ej: "Spain", "Wifi", "Apartment"), traduce los filtros del usuario al inglés cuando sea necesario."""),
        ("human", "{pregunta}")
    ])

    cadena_query = prompt_generador | llm
    respuesta_llm = cadena_query.invoke({
        "esquema": ESQUEMA_AIRBNB,
        "pregunta": pregunta_usuario
    })

    texto_pipeline = respuesta_llm.content.strip()
    if texto_pipeline.startswith("```"):
        texto_pipeline = texto_pipeline.replace("```json", "").replace("```", "").strip()

    print("1. Consulta MongoDB (Pipeline) generada:")
    print(texto_pipeline)

    # 5. Ejecutar en MongoDB Atlas
    try:
        pipeline = json.loads(texto_pipeline)
        resultados_crudos = list(coleccion.aggregate(pipeline))
        resultados_json = json_util.dumps(resultados_crudos, ensure_ascii=False)
    except Exception as e:
        print(f"\nError al ejecutar la consulta en MongoDB: {e}")
        return

    print(f"\n2. Datos recuperados de Atlas ({len(resultados_crudos)} registros):")
    print(resultados_json)

    # 6. Prompt 2: Redactar respuesta final en español
    prompt_respuesta = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Responde en español de forma clara y directa a la pregunta del usuario basándote ÚNICAMENTE en los datos obtenidos de la base de datos. Si los datos están vacíos ([]), indica que no se encontraron resultados."),
        ("human", "Pregunta original: {pregunta}\n\nDatos obtenidos de MongoDB: {datos}")
    ])

    cadena_final = prompt_respuesta | llm
    respuesta_final = cadena_final.invoke({
        "pregunta": pregunta_usuario,
        "datos": resultados_json
    })

    print("\n3. Respuesta final del sistema:")
    print(respuesta_final.content)
    print("=" * 60)

def imprimir_banner_modelo(nombre_modelo: str):
    """Imprime un separador visual grande en la terminal para cada modelo."""
    borde = "█" * 80
    print(f"\n\n{borde}")
    print(f"██  EVALUANDO MODELO: {nombre_modelo.upper()}")
    print(f"{borde}")

def imprimir_banner_pregunta(numero: int, total: int, pregunta: str):
    """Imprime un separador visual claro para cada pregunta."""
    linea = "═" * 80
    print(f"\n{linea}")
    print(f"▶ PREGUNTA [{numero}/{total}]: {pregunta}")
    print(f"{linea}")

def consultar_airbnb_con_metricas(pregunta_usuario: str, modelo: str) -> dict:
    """
    Ejecuta una pregunta en lenguaje natural contra MongoDB Atlas usando el modelo indicado
    y devuelve un diccionario con las métricas de rendimiento.
    """
    inicio_total = time.time()
    
    # Estructura para guardar las métricas de esta ejecución
    metricas = {
        "Modelo": modelo,
        "Pregunta": pregunta_usuario[:40] + "...",
        "Exito_Consulta": False,
        "Registros_Hallados": 0,
        "Tiempo_LLM_Query (s)": 0.0,
        "Tiempo_Total (s)": 0.0,
        "Tokens_Totales": 0,
        "Error": "Ninguno"
    }

    cliente = MongoClient(mongo_uri)
    coleccion = cliente["sample_airbnb"]["listingsAndReviews"]
    llm = ChatGroq(model=modelo, temperature=0)

    # --- FASE 1: GENERACIÓN DE LA CONSULTA MONGODB ---
    prompt_generador = ChatPromptTemplate.from_messages([
        ("system", """Eres un experto en bases de datos MongoDB. Tu única tarea es traducir la pregunta del usuario en un Pipeline de Agregación válido de MongoDB (una lista JSON de etapas como $match, $group, $sort, $project, $limit).

Usa EXCLUSIVAMENTE este esquema:
{esquema}

REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el array JSON válido. NO incluyas bloques de código markdown (```json), ni explicaciones, ni texto adicional.
2. Si buscas alojamientos concretos (no conteos ni medias), incluye SIEMPRE una etapa "$project" seleccionando solo los campos relevantes (y excluyendo "_id": 0) y un "$limit": 5 para no devolver documentos gigantes.
3. Los valores de texto en la base de datos están en inglés (ej: "Spain", "Wifi", "Apartment"), traduce los filtros del usuario al inglés cuando sea necesario."""),
        ("human", "{pregunta}")
    ])

    cadena_query = prompt_generador | llm
    
    inicio_llm_1 = time.time()
    try:
        respuesta_llm = cadena_query.invoke({
            "esquema": ESQUEMA_AIRBNB,
            "pregunta": pregunta_usuario
        })
    except Exception as e:
        metricas["Error"] = f"Fallo API Groq: {str(e)[:50]}"
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)
        print(f"\n[ERROR DE API]: {e}")
        return metricas

    metricas["Tiempo_LLM_Query (s)"] = round(time.time() - inicio_llm_1, 2)
    
    # Extraemos el consumo de tokens de la primera llamada
    tokens_1 = respuesta_llm.response_metadata.get("token_usage", {}).get("total_tokens", 0)
    metricas["Tokens_Totales"] += tokens_1

    texto_pipeline = respuesta_llm.content.strip()
    if texto_pipeline.startswith("```"):
        texto_pipeline = texto_pipeline.replace("```json", "").replace("```", "").strip()

    print("\n[1] CONSULTA MONGODB GENERADA:")
    print(texto_pipeline)

    # --- FASE 2: EJECUCIÓN EN MONGODB ATLAS ---
    try:
        pipeline = json.loads(texto_pipeline)
        resultados_crudos = list(coleccion.aggregate(pipeline))
        resultados_json = json_util.dumps(resultados_crudos, ensure_ascii=False)
        
        metricas["Exito_Consulta"] = True
        metricas["Registros_Hallados"] = len(resultados_crudos)
    except Exception as e:
        metricas["Error"] = f"Error Mongo/JSON: {str(e)[:50]}"
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)
        print(f"\n[ERROR AL EJECUTAR EN MONGODB]: {e}")
        return metricas

    print(f"\n[2] DATOS RECUPERADOS DE ATLAS ({len(resultados_crudos)} registros):")
    print(resultados_json)

    # --- FASE 3: REDACCIÓN DE RESPUESTA HUMANA ---
    prompt_respuesta = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Responde en español de forma clara y directa a la pregunta del usuario basándote ÚNICAMENTE en los datos obtenidos de la base de datos. Si los datos están vacíos ([]), indica que no se encontraron resultados."),
        ("human", "Pregunta original: {pregunta}\n\nDatos obtenidos de MongoDB: {datos}")
    ])

    cadena_final = prompt_respuesta | llm
    respuesta_final = cadena_final.invoke({
        "pregunta": pregunta_usuario,
        "datos": resultados_json
    })

    tokens_2 = respuesta_final.response_metadata.get("token_usage", {}).get("total_tokens", 0)
    metricas["Tokens_Totales"] += tokens_2
    metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)

    print("\n[3] RESPUESTA FINAL:")
    print(respuesta_final.content)
    print(f"\n⏱️  Métricas: Consulta LLM: {metricas['Tiempo_LLM_Query (s)']}s | Total: {metricas['Tiempo_Total (s)']}s | Tokens: {metricas['Tokens_Totales']}")

    return metricas

if __name__ == "__main__":
    # probar_conexion_airbnb()

    # =========================================================================
    # ZONA DE CONFIGURACIÓN: Añade o quita aquí los modelos y preguntas que quieras
    # =========================================================================
    
    MIS_MODELOS = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b"
    ]

    MIS_PREGUNTAS = [
        # Nivel 1: Filtro simple y ordenación
        "Dime 3 apartamentos en Barcelona que tengan Wifi y aire acondicionado, ordenados del más barato al más caro.",
        
        # Nivel 2: Subdocumentos anidados y múltiples condiciones numéricas
        "Dime 3 alojamientos en España donde el anfitrión sea Superhost, tengan capacidad para al menos 4 personas y una puntuación global superior a 95.",
        
        # Nivel 3: Agregación estadística ($group y $avg)
        "¿Cuál es el precio medio por noche de los alojamientos en Barcelona según el tipo de propiedad (property_type)? Ordénalo de más caro a más barato."
    ]

    # =========================================================================
    # EJECUCIÓN DE LA BATERÍA DE PRUEBAS
    # =========================================================================
    historial_metricas = []

    for modelo in MIS_MODELOS:
        imprimir_banner_modelo(modelo)
        
        for idx, pregunta in enumerate(MIS_PREGUNTAS, start=1):
            imprimir_banner_pregunta(idx, len(MIS_PREGUNTAS), pregunta)
            resultado_metrica = consultar_airbnb_con_metricas(pregunta, modelo)
            historial_metricas.append(resultado_metrica)
            
            # Pequeña pausa de 2 segundos entre consultas para respetar el límite por minuto de Groq
            time.sleep(2)

    # =========================================================================
    # TABLA COMPARATIVA FINAL
    # =========================================================================
    print("\n\n" + "★" * 80)
    print("RESUMEN COMPARATIVO DE RENDIMIENTO (MONGODB ATLAS + API)")
    print("★" * 80)
    df_resultados = pd.DataFrame(historial_metricas)
    print(df_resultados.to_string(index=False))
