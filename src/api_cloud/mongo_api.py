import os
import json
import time
import pandas as pd
from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# 1. CONFIGURACIÓN DE CREDENCIALES Y PARÁMETROS GLOBALES
# =============================================================================
# Cargamos las variables privadas desde el archivo .env situado en la raíz
load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
groq_api_key = os.getenv("GROQ_API_KEY")

# Modelo por defecto configurable mediante variable de entorno
MODELO_POR_DEFECTO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Esquema descriptivo (Schema Grounding):
# Como MongoDB no tiene un esquema rígido (es schemaless), proporcionamos al LLM
# un mapa exacto de los campos, tipos de datos y subdocumentos anidados para evitar
# que invente nombres de atributos inexistentes al construir las consultas.
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

# =============================================================================
# 2. FUNCIONES DE DIAGNÓSTICO Y UTILIDADES VISUALES
# =============================================================================
def probar_conexion_airbnb():
    """Valida la conexión con MongoDB Atlas y recupera un documento de muestra."""
    print("Conectando a MongoDB Atlas (sample_airbnb)...")
    
    if not mongo_uri:
        print("ERROR: No se encontró la variable MONGODB_URI en el archivo .env")
        return

    cliente = MongoClient(mongo_uri)
    db = cliente["sample_airbnb"]
    coleccion = db["listingsAndReviews"]
    
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

def imprimir_banner_modelo(nombre_modelo: str):
    """Imprime un separador visual grande en la terminal para identificar cada modelo."""
    borde = "█" * 80
    print(f"\n\n{borde}")
    print(f"██  EVALUANDO MODELO: {nombre_modelo.upper()}")
    print(f"{borde}")

def imprimir_banner_pregunta(numero: int, total: int, pregunta: str):
    """Imprime un separador visual estructurado para cada pregunta evaluada."""
    linea = "═" * 80
    print(f"\n{linea}")
    print(f"▶ PREGUNTA [{numero}/{total}]: {pregunta}")
    print(f"{linea}")

# =============================================================================
# 3. MOTOR PRINCIPAL DE CONSULTA TEXT-TO-NOSQL CON TELEMETRÍA
# =============================================================================
def consultar_airbnb_con_metricas(pregunta_usuario: str, modelo: str) -> dict:
    """
    Traduce una pregunta en lenguaje natural a un Pipeline de Agregación de MongoDB,
    lo ejecuta en Atlas, sintetiza la respuesta en español y devuelve métricas de rendimiento.
    """
    inicio_total = time.time()
    
    # Diccionario de telemetría para comparar precisión, latencia y consumo de tokens
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

    # Inicializamos el cliente NoSQL en la nube y el modelo LLM vía API
    # temperature=0 elimina la aleatoriedad para garantizar salidas JSON estructuradas
    cliente = MongoClient(mongo_uri)
    coleccion = cliente["sample_airbnb"]["listingsAndReviews"]
    llm = ChatGroq(model=modelo, temperature=0)

    # -------------------------------------------------------------------------
    # FASE 1: TRADUCCIÓN DE LENGUAJE NATURAL A AGGREGATION PIPELINE (TEXT-TO-MQL)
    # -------------------------------------------------------------------------
    # Definimos el rol del sistema, las reglas de formato JSON y la traducción de filtros
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

    # Encadenamos la plantilla de prompt con el modelo mediante LCEL (operador |)
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

    # Calculamos la latencia pura del modelo generando la consulta y sumamos tokens
    metricas["Tiempo_LLM_Query (s)"] = round(time.time() - inicio_llm_1, 2)
    tokens_1 = respuesta_llm.response_metadata.get("token_usage", {}).get("total_tokens", 0)
    metricas["Tokens_Totales"] += tokens_1

    # Limpieza preventiva: eliminamos delimitadores Markdown si el modelo los incluyó
    texto_pipeline = respuesta_llm.content.strip()
    if texto_pipeline.startswith("```"):
        texto_pipeline = texto_pipeline.replace("```json", "").replace("```", "").strip()

    print("\n[1] CONSULTA MONGODB GENERADA:")
    print(texto_pipeline)

    # -------------------------------------------------------------------------
    # FASE 2: VALIDACIÓN JSON Y EJECUCIÓN EN EL CLÚSTER DE MONGODB ATLAS
    # -------------------------------------------------------------------------
    try:
        # Convertimos el string JSON generado por el LLM a una lista nativa de Python
        pipeline = json.loads(texto_pipeline)
        
        # Ejecutamos el pipeline directamente en los servidores de Atlas
        resultados_crudos = list(coleccion.aggregate(pipeline))
        
        # Usamos bson.json_util en lugar de json estándar porque MongoDB devuelve tipos
        # propios como Decimal128 (en los precios) que provocarían error en json.dumps()
        resultados_json = json_util.dumps(resultados_crudos, ensure_ascii=False)
        
        metricas["Exito_Consulta"] = True
        metricas["Registros_Hallados"] = len(resultados_crudos)
    except Exception as e:
        # Si el LLM generó un JSON inválido o un operador erróneo, lo capturamos aquí
        metricas["Error"] = f"Error Mongo/JSON: {str(e)[:50]}"
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)
        print(f"\n[ERROR AL EJECUTAR EN MONGODB]: {e}")
        return metricas

    print(f"\n[2] DATOS RECUPERADOS DE ATLAS ({len(resultados_crudos)} registros):")
    print(resultados_json)

    # -------------------------------------------------------------------------
    # FASE 3: SÍNTESIS DE RESPUESTA HUMANA FUNDAMENTADA EN LOS DATOS (GROUNDING)
    # -------------------------------------------------------------------------
    # Segunda llamada al LLM: transforma el JSON crudo en una respuesta clara en español,
    # restringiendo el contexto únicamente a los registros reales recuperados.
    prompt_respuesta = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Responde en español de forma clara y directa a la pregunta del usuario basándote ÚNICAMENTE en los datos obtenidos de la base de datos. Si los datos están vacíos ([]), indica que no se encontraron resultados."),
        ("human", "Pregunta original: {pregunta}\n\nDatos obtenidos de MongoDB: {datos}")
    ])

    cadena_final = prompt_respuesta | llm
    respuesta_final = cadena_final.invoke({
        "pregunta": pregunta_usuario,
        "datos": resultados_json
    })

    # Contabilizamos los tokens de la segunda llamada y el tiempo total de extremo a extremo
    tokens_2 = respuesta_final.response_metadata.get("token_usage", {}).get("total_tokens", 0)
    metricas["Tokens_Totales"] += tokens_2
    metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)

    print("\n[3] RESPUESTA FINAL:")
    print(respuesta_final.content)
    print(f"\n⏱️  Métricas: Consulta LLM: {metricas['Tiempo_LLM_Query (s)']}s | Total: {metricas['Tiempo_Total (s)']}s | Tokens: {metricas['Tokens_Totales']}")

    return metricas

# =============================================================================
# 4. BATERÍA DE EVALUACIÓN COMPARATIVA
# =============================================================================
if __name__ == "__main__":
    MIS_MODELOS = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b"
    ]

    MIS_PREGUNTAS = [
        # Nivel 1: Filtrado en arrays ($all) y ordenación ($sort)
        "Dime 3 apartamentos en Barcelona que tengan Wifi y aire acondicionado, ordenados del más barato al más caro.",
        
        # Nivel 2: Navegación por subdocumentos anidados y comparadores numéricos ($gte, $gt)
        "Dime 3 alojamientos en España donde el anfitrión sea Superhost, tengan capacidad para al menos 4 personas y una puntuación global superior a 95.",
        
        # Nivel 3: Pipeline de agregación estadística ($group y $avg)
        "¿Cuál es el precio medio por noche de los alojamientos en Barcelona según el tipo de propiedad (property_type)? Ordénalo de más caro a más barato."
    ]

    historial_metricas = []

    for modelo in MIS_MODELOS:
        imprimir_banner_modelo(modelo)
        
        for idx, pregunta in enumerate(MIS_PREGUNTAS, start=1):
            imprimir_banner_pregunta(idx, len(MIS_PREGUNTAS), pregunta)
            resultado_metrica = consultar_airbnb_con_metricas(pregunta, modelo)
            historial_metricas.append(resultado_metrica)
            
            # Pausa preventiva para no exceder el límite de tokens/peticiones por minuto de Groq
            time.sleep(2)

    # Consolidamos todas las ejecuciones en una tabla comparativa con Pandas
    print("\n\n" + "★" * 80)
    print("RESUMEN COMPARATIVO DE RENDIMIENTO (MONGODB ATLAS + API)")
    print("★" * 80)
    df_resultados = pd.DataFrame(historial_metricas)
    print(df_resultados.to_string(index=False))