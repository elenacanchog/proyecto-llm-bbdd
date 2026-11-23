import os
import time
import pandas as pd
from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# =============================================================================
# 1. CONFIGURACIÓN DE CREDENCIALES Y CONEXIÓN A MONGODB ATLAS
# =============================================================================
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise ValueError("ERROR: No se encontró la variable MONGODB_URI en el archivo .env")

cliente = MongoClient(MONGO_URI)
mongo_db = cliente["sample_airbnb"]

CAMPOS_PESADOS = {
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

SYSTEM_PROMPT = """
Eres un agente experto en bases de datos MongoDB.
Responde siempre en español de forma clara y directa usando exclusivamente los datos obtenidos de la base de datos.

Reglas obligatorias:
1. Llama SIEMPRE primero a get_mongo_schema para ver qué colecciones existen y los nombres exactos de sus campos.
2. Los valores de texto en la base de datos están en inglés: traduce los filtros del usuario al inglés (ej. "Spain", "Apartment", "Wifi", "Air conditioning").
3. Para filtrar, ordenar, limitar resultados o calcular medias/agrupaciones, usa operacion='aggregate' pasando en consulta_json un string con una LISTA JSON de etapas: '[{"$match": ...}, {"$sort": ...}, {"$limit": 5}]'.
4. Si incluyes "$project", proyecta siempre los campos usados en "$match" y subcampos concretos (ej. "host.host_name": 1).
5. Si una consulta no devuelve resultados o hay un error de sintaxis, revisa el esquema y reintenta corrigiendo la consulta.
"""

# =============================================================================
# 2. UTILIDADES DE INSPECCIÓN DE ESQUEMA Y HERRAMIENTAS DEL AGENTE
# =============================================================================
def resumir_valor(valor, nivel: int = 0):
    """Recorta textos y arrays largos de un documento de muestra para mostrar un esquema compacto."""
    if isinstance(valor, dict):
        if nivel >= 2:
            return "{...}"
        return {k: resumir_valor(v, nivel + 1) for k, v in valor.items()}
    if isinstance(valor, list):
        if not valor:
            return []
        muestra = [resumir_valor(x, nivel + 1) for x in valor[:12]]
        if len(valor) > 12:
            muestra.append(f"... (+{len(valor) - 12})")
        return muestra
    if isinstance(valor, str) and len(valor) > 35:
        return valor[:35] + "..."
    return valor

@tool
def get_mongo_schema() -> str:
    """Devuelve las colecciones disponibles en MongoDB y un documento de ejemplo compacto con sus campos."""
    colecciones = mongo_db.list_collection_names()
    if not colecciones:
        return "No hay colecciones en esta base de datos."

    resultado = []
    for nombre_col in colecciones:
        doc = mongo_db[nombre_col].find_one({}, CAMPOS_PESADOS)
        resumen = resumir_valor(doc) if doc else {}
        # Sin indent=2 para ahorrar cientos de tokens de espacios en blanco
        resultado.append(
            f"Colección '{nombre_col}': "
            + json_util.dumps(resumen, ensure_ascii=False)
        )
    return "\n".join(resultado)

@tool
def run_mongo(coleccion: str, operacion: str, consulta_json: str = "{}") -> str:
    """
    Ejecuta una consulta de lectura en MongoDB Atlas sobre una colección.
    - coleccion: nombre de la colección (ej. 'listingsAndReviews').
    - operacion: 'find', 'aggregate' o 'count'.
    - consulta_json: string JSON con el filtro (en find/count) o lista JSON de etapas (en aggregate).
    """
    try:
        col = mongo_db[coleccion]
        consulta = json_util.loads(consulta_json) if isinstance(consulta_json, str) else consulta_json

        if operacion == "find":
            filas = list(col.find(consulta, CAMPOS_PESADOS).limit(5))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "No se encontraron documentos."

        elif operacion == "aggregate":
            if isinstance(consulta, dict):
                consulta = [{k: v} for k, v in consulta.items()]

            tiene_proyeccion_o_grupo = any(
                any(k in etapa for k in ("$project", "$group", "$count"))
                for etapa in consulta if isinstance(etapa, dict)
            )
            pipeline = consulta if tiene_proyeccion_o_grupo else consulta + [{"$project": CAMPOS_PESADOS}]

            filas = list(col.aggregate(pipeline))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "La agregación no devolvió resultados."

        elif operacion == "count":
            return f"Total de documentos: {col.count_documents(consulta)}"

        else:
            return "Operación no válida. Usa únicamente: 'find', 'aggregate' o 'count'."

    except Exception as e:
        return f"Error MongoDB: {e}"

HERRAMIENTAS_MONGO = [get_mongo_schema, run_mongo]

# =============================================================================
# 3. UTILIDADES VISUALES PARA LA CONSOLA
# =============================================================================
def imprimir_banner_modelo(nombre_modelo: str):
    borde = "█" * 80
    print(f"\n\n{borde}")
    print(f"██  EVALUANDO MODELO (MONGODB API): {nombre_modelo.upper()}")
    print(f"{borde}")

def imprimir_banner_pregunta(numero: int, total: int, pregunta: str):
    linea = "═" * 80
    print(f"\n{linea}")
    print(f"▶ PREGUNTA [{numero}/{total}]: {pregunta}")
    print(f"{linea}")

# =============================================================================
# 4. MOTOR PRINCIPAL DEL AGENTE TEXT-TO-NOSQL CON TELEMETRÍA
# =============================================================================
def consultar_airbnb_con_metricas(pregunta_usuario: str, modelo: str) -> dict:
    inicio_total = time.time()

    metricas = {
        "Modelo": modelo,
        "Pregunta": pregunta_usuario[:40] + "...",
        "Exito_Consulta": False,
        "Intentos_Mongo": 0,
        "Tiempo_Total (s)": 0.0,
        "Tokens_Totales": 0,
        "Error": "Ninguno"
    }

    try:
        # max_tokens=800 evita que Groq rechace modelos con límite OTPM de 1000 (como Qwen)
        llm = ChatGroq(model=modelo, temperature=0, max_tokens=800)

        agente = create_agent(
            model=llm,
            tools=HERRAMIENTAS_MONGO,
            system_prompt=SYSTEM_PROMPT,
        )

        resultado_agente = agente.invoke(
            {"messages": [("user", pregunta_usuario)]},
            config={"recursion_limit": 10}
        )
        mensajes = resultado_agente["messages"]

        consulta_ejecutada = "No se ejecutó ninguna operación en MongoDB"
        datos_crudos = "Sin datos"
        hubo_fallo_previo = False

        for msg in mensajes:
            if hasattr(msg, "response_metadata") and msg.response_metadata:
                uso_tokens = msg.response_metadata.get("token_usage", {})
                metricas["Tokens_Totales"] += uso_tokens.get("total_tokens", 0)

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for llamada in msg.tool_calls:
                    if llamada["name"] == "run_mongo":
                        metricas["Intentos_Mongo"] += 1
                        op = llamada["args"].get("operacion", "")
                        q_json = llamada["args"].get("consulta_json", "")
                        consulta_ejecutada = f"[{op}] {q_json}"

            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "run_mongo":
                datos_crudos = msg.content
                if not datos_crudos.startswith("Error MongoDB:") and not datos_crudos.startswith("Operación no válida"):
                    metricas["Exito_Consulta"] = True
                    metricas["Error"] = "Corregido tras reintento" if hubo_fallo_previo else "Ninguno"
                else:
                    hubo_fallo_previo = True
                    metricas["Error"] = datos_crudos[:50]

        respuesta_final = mensajes[-1].content
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)

        print("\n[1] CONSULTA MONGODB GENERADA Y EJECUTADA POR EL AGENTE:")
        print(consulta_ejecutada)

        print("\n[2] DATOS DEVUELTOS POR MONGODB ATLAS:")
        print(datos_crudos)

        print("\n[3] RESPUESTA FINAL:")
        print(respuesta_final)
        print(f"\n⏱️  Métricas: Intentos Mongo: {metricas['Intentos_Mongo']} | Tiempo Total: {metricas['Tiempo_Total (s)']}s | Tokens: {metricas['Tokens_Totales']}")

    except Exception as e:
        metricas["Error"] = f"Fallo Agente/API: {str(e)[:50]}"
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)
        print(f"\n[ERROR CRÍTICO]: {e}")

    return metricas

# =============================================================================
# 5. BATERÍA DE EVALUACIÓN COMPARATIVA
# =============================================================================
if __name__ == "__main__":
    MIS_MODELOS = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b"
    ]

    MIS_PREGUNTAS = [
        "Dime 3 apartamentos en Barcelona que tengan Wifi y aire acondicionado, ordenados del más barato al más caro.",
        "Dime 3 alojamientos en España donde el anfitrión sea Superhost, tengan capacidad para al menos 4 personas y una puntuación global superior a 95.",
        "¿Cuál es el precio medio por noche de los alojamientos en Barcelona según el tipo de propiedad (property_type)? Ordénalo de más caro a más barato."
    ]

    historial_metricas = []

    for modelo in MIS_MODELOS:
        imprimir_banner_modelo(modelo)
        
        for idx, pregunta in enumerate(MIS_PREGUNTAS, start=1):
            imprimir_banner_pregunta(idx, len(MIS_PREGUNTAS), pregunta)
            resultado_metrica = consultar_airbnb_con_metricas(pregunta, modelo)
            historial_metricas.append(resultado_metrica)
            # Pausa de 6s para evitar saturar el límite de tokens de salida por minuto en Groq
            time.sleep(6)

    print("\n\n" + "★" * 80)
    print("RESUMEN COMPARATIVO DE RENDIMIENTO (MONGODB ATLAS + AGENTE API)")
    print("★" * 80)
    df_resultados = pd.DataFrame(historial_metricas)
    print(df_resultados.to_string(index=False))