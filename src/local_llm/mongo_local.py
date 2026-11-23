import os
import threading
import time
from typing import Any
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import json_util
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()

# 1. CONFIGURACIÓN INICIAL
URI_MONGO = os.getenv("LOCAL_MONGO_URI")
NOMBRE_BD = os.getenv("LOCAL_MONGO_DB")
MODELOS_A_PROBAR = ["qwen3.5:4b", "gemma-4:latest"]

cliente = MongoClient(URI_MONGO)
mongo_db = cliente[NOMBRE_BD]

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
    "host": 0,
}

SYSTEM_PROMPT = """
Eres un agente experto en MongoDB. 
Responde en español usando exclusivamente los datos de la base de datos.

Reglas obligatorias:
1. Llama SIEMPRE primero a obtener_esquema_mongo para ver los nombres reales de todos los campos (ej. "address.market", "property_type", "amenities", "price").
2. Traduce los valores de búsqueda a como aparecen en la BD (ej. "Apartment", "Wifi", "Air conditioning").
3. Para filtrar, ordenar y limitar resultados (Top N), usa operacion='aggregate' con una LISTA JSON de etapas: [{"$match": ...}, {"$sort": ...}, {"$limit": 3}].
4. Si una consulta devuelve "No se encontraron documentos", revisa los nombres de los campos en el esquema antes de reintentar.
5. Incluye SIEMPRE en tu respuesta final la consulta o pipeline JSON exacto que ejecutaste.
"""


# 2. UTILIDADES DE TIEMPO, SPINNER Y ESQUEMA COMPACTO
def formatear_tiempo(segundos: float) -> str:
    total_segundos = int(round(segundos))
    minutos = total_segundos // 60
    segs = total_segundos % 60
    if minutos == 0:
        return str(round(segundos, 2)) + " segundos"
    texto_min = "1 minuto" if minutos == 1 else str(minutos) + " minutos"
    texto_seg = "1 segundo" if segs == 1 else str(segs) + " segundos"
    return texto_min + " " + texto_seg


class Spinner:
    def __init__(self, texto: str = "Pensando"):
        self.texto = texto
        self._detener = threading.Event()
        self._hilo = None

    def _girar(self):
        simbolos = ["|", "/", "-", "\\"]
        i = 0
        while not self._detener.is_set():
            print("\r" + self.texto + " " + simbolos[i % 4], end="", flush=True)
            i += 1
            time.sleep(0.15)

    def iniciar(self):
        self._detener.clear()
        self._hilo = threading.Thread(target=self._girar, daemon=True)
        self._hilo.start()

    def detener(self):
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join()
        print("\r" + " " * (len(self.texto) + 5) + "\r", end="", flush=True)


def resumir_valor(valor, nivel=0):
    """Muestra TODAS las claves principales sin recortarlas, pero acorta textos y listas largas."""
    if isinstance(valor, dict):
        if nivel >= 2:
            return "{...}"
        return {k: resumir_valor(v, nivel + 1) for k, v in valor.items()}
    if isinstance(valor, list):
        if not valor:
            return []
        muestra = [resumir_valor(x, nivel + 1) for x in valor[:6]]
        if len(valor) > 6:
            muestra.append("... (+" + str(len(valor) - 6) + " elementos)")
        return muestra
    if isinstance(valor, str) and len(valor) > 40:
        return valor[:40] + "..."
    return valor


# 3. HERRAMIENTAS MONGODB
@tool
def obtener_esquema_mongo() -> str:
    """Devuelve las colecciones disponibles y todos sus campos con valores de ejemplo."""
    colecciones = mongo_db.list_collection_names()
    if not colecciones:
        return "No hay colecciones en esta base de datos."

    resultado = []
    for nombre_col in colecciones:
        doc = mongo_db[nombre_col].find_one({}, CAMPOS_PESADOS)
        resumen = resumir_valor(doc) if doc else {}
        resultado.append(
            "Colección '" + nombre_col + "' | Campos y ejemplo:\n"
            + json_util.dumps(resumen, indent=2, ensure_ascii=False)
        )
    return "\n\n".join(resultado)


@tool
def ejecutar_mongo(
    coleccion: str,
    operacion: str,
    consulta_json: Any = "{}",
    datos_json: Any = "{}"
) -> str:
    """
    Ejecuta una operación en MongoDB sobre una colección.
    - coleccion: nombre de la colección.
    - operacion: 'find', 'aggregate', 'count', 'insert_one', 'update_many' o 'delete_many'.
    - consulta_json: filtro JSON (en find/count/update/delete) o lista JSON de etapas (en aggregate).
    - datos_json: documento a insertar o cambios ($set) para update_many.
    """
    try:
        col = mongo_db[coleccion]
        consulta = json_util.loads(consulta_json) if isinstance(consulta_json, str) else consulta_json
        datos = json_util.loads(datos_json) if isinstance(datos_json, str) else datos_json

        if operacion == "find":
            filas = list(col.find(consulta, CAMPOS_PESADOS).limit(5))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "No se encontraron documentos."

        elif operacion == "aggregate":
            # Si el LLM pasa un diccionario único en lugar de una lista de etapas, lo corregimos
            if isinstance(consulta, dict):
                consulta = [{k: v} for k, v in consulta.items()]

            tiene_project = any("$project" in etapa for etapa in consulta if isinstance(etapa, dict))
            pipeline = consulta if tiene_project else consulta + [{"$project": CAMPOS_PESADOS}]

            filas = list(col.aggregate(pipeline))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "La agregación no devolvió resultados."

        elif operacion == "count":
            return "Total de documentos: " + str(col.count_documents(consulta))

        elif operacion == "insert_one":
            res = col.insert_one(datos)
            return "Documento insertado con ID: " + str(res.inserted_id)

        elif operacion == "update_many":
            res = col.update_many(consulta, datos)
            return "Documentos actualizados: " + str(res.modified_count)

        elif operacion == "delete_many":
            res = col.delete_many(consulta)
            return "Documentos eliminados: " + str(res.deleted_count)

        else:
            return "Operación no válida. Usa: find, aggregate, count, insert_one, update_many o delete_many."

    except Exception as e:
        return "Error en MongoDB: " + str(e)


tools = [obtener_esquema_mongo, ejecutar_mongo]


# 4. EJECUCIÓN Y COMPARATIVA DE MODELOS
def probar_modelo(nombre_modelo: str, pregunta: str):
    print("\n" + "=" * 60)
    print("PROBANDO MODELO (MongoDB):", nombre_modelo)
    print("=" * 60)

    # Desactivamos el pensamiento largo en Qwen3 para acelerar la respuesta
    #extra_params = {"reasoning": False} if "qwen3" in nombre_modelo.lower() else {}

    llm = ChatOllama(
        model=nombre_modelo,
        temperature=0,
        num_ctx=8192,
        keep_alive="15m",
        #**extra_params,
    )

    agente = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    spinner = Spinner("Pensando")
    inicio = time.perf_counter()
    spinner.iniciar()

    try:
        # recursion_limit=10 evita bucles infinitos si un modelo se atasca
        for paso in agente.stream(
            {"messages": [("user", pregunta)]},
            stream_mode="values",
            config={"recursion_limit": 10},
        ):
            msg = paso["messages"][-1]
            if msg.type == "human":
                continue

            spinner.detener()

            if msg.type == "ai" and msg.tool_calls:
                for llamada in msg.tool_calls:
                    print("[Herramienta] ->", llamada["name"], "| Args:", llamada["args"])
            elif msg.type == "tool":
                contenido = str(msg.content)
                vista_previa = contenido[:140] + "..." if len(contenido) > 140 else contenido
                print("[BD respondió] ->", vista_previa)
            elif msg.type == "ai" and msg.content:
                print("\n--- RESPUESTA FINAL ---")
                print(msg.content)

            spinner.iniciar()
    finally:
        spinner.detener()

    duracion = time.perf_counter() - inicio
    print("\nTiempo total (" + nombre_modelo + "):", formatear_tiempo(duracion))


if __name__ == "__main__":
    pregunta_prueba = "Dime 3 apartamentos en Barcelona que tengan Wifi y aire acondicionado, ordenados del más barato al más caro."

    for modelo in MODELOS_A_PROBAR:
        try:
            probar_modelo(modelo, pregunta_prueba)
        except Exception as e:
            print("\nError ejecutando", modelo + ":", e)