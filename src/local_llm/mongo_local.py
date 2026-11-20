import threading
import time
from pymongo import MongoClient
from bson import json_util
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# 1. CONFIGURACIÓN INICIAL
URI_MONGO = "mongodb://localhost:27017/"
NOMBRE_BD = "sampl_airbnb" 
MODELOS_A_PROBAR = ["qwen3.5:4b", "gemma-4:latest"]

cliente = MongoClient(URI_MONGO)
mongo_db = cliente[NOMBRE_BD]

SYSTEM_PROMPT = """
Eres un agente experto en bases de datos NoSQL con MongoDB.
Dada una petición del usuario, construye la consulta o pipeline en formato JSON estricto,
ejecútala y responde usando los resultados obtenidos.

Reglas obligatorias:
1. Revisa SIEMPRE primero qué colecciones y campos existen con obtener_esquema_mongo antes de consultar.
2. Usa comillas dobles en todas las claves y operadores JSON (ej. "$gt", "$match", "$group", "$sort", "$limit").
3. Para búsquedas simples usa operacion='find'. Para ordenar, agrupar, calcular promedios o sacar un Top N, usa operacion='aggregate' pasando una lista JSON de etapas.
4. Puedes ejecutar find, aggregate, count, insert_one, update_many y delete_many cuando el usuario lo solicite.
5. Incluye SIEMPRE en tu respuesta final la consulta o pipeline JSON exacto que ejecutaste.
"""


# 2. UTILIDADES DE TIEMPO Y CARGA EN CONSOLA
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
    def __init__(self, texto: str = "Ejecutando"):
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


# 3. HERRAMIENTAS MONGODB (LECTURA Y ESCRITURA)
@tool
def obtener_esquema_mongo() -> str:
    """Devuelve las colecciones disponibles en MongoDB y 2 documentos de ejemplo de cada una para conocer su estructura."""
    colecciones = mongo_db.list_collection_names()
    if not colecciones:
        return "No hay colecciones en esta base de datos."

    resultado = []
    for nombre_col in colecciones:
        muestras = list(mongo_db[nombre_col].find({}, {"_id": 0}).limit(2))
        resultado.append(
            "Colección '" + nombre_col + "' | Documentos de ejemplo:\n"
            + json_util.dumps(muestras, indent=2, ensure_ascii=False)
        )
    return "\n\n".join(resultado)


@tool
def ejecutar_mongo(
    coleccion: str,
    operacion: str,
    consulta_json: str = "{}",
    datos_json: str = "{}"
) -> str:
    """
    Ejecuta una operación en MongoDB sobre una colección.
    - coleccion: nombre de la colección objetivo.
    - operacion: 'find', 'aggregate', 'count', 'insert_one', 'update_many' o 'delete_many'.
    - consulta_json: filtro JSON (ej. '{"campo": {"$gt": 100}}') o lista JSON de etapas si operacion='aggregate'.
    - datos_json: documento JSON a insertar (en insert_one) o actualización JSON (ej. '{"$set": {"campo": "valor"}}' en update_many).
    """
    try:
        col = mongo_db[coleccion]

        # Acepta tanto texto JSON como diccionarios/listas si el modelo los pasa ya parseados
        consulta = json_util.loads(consulta_json) if isinstance(consulta_json, str) else consulta_json
        datos = json_util.loads(datos_json) if isinstance(datos_json, str) else datos_json

        if operacion == "find":
            filas = list(col.find(consulta, {"_id": 0}).limit(5))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "No se encontraron documentos."

        elif operacion == "aggregate":
            filas = list(col.aggregate(consulta))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "La agregación no devolvió resultados."

        elif operacion == "count":
            return "Total de documentos: " + str(col.count_documents(consulta))

        elif operacion == "insert_one":
            res = col.insert_one(datos)
            return "Documento insertado correctamente con ID: " + str(res.inserted_id)

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


# 4. FUNCIÓN PARA PROBAR Y CRONOMETRAR CADA MODELO
def probar_modelo(nombre_modelo: str, pregunta: str):
    print("\n" + "=" * 60)
    print("PROBANDO MODELO (MongoDB):", nombre_modelo)
    print("=" * 60)

    llm = ChatOllama(
        model=nombre_modelo,
        temperature=0,
        keep_alive="15m",
    )

    agente = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    spinner = Spinner("Ejecutando")
    inicio = time.perf_counter()
    spinner.iniciar()

    try:
        for paso in agente.stream({"messages": [("user", pregunta)]}, stream_mode="values"):
            spinner.detener()
            paso["messages"][-1].pretty_print()
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