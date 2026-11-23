import os
import threading
import time
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()

# 1. CONFIGURACIÓN INICIAL
URI_BD = os.getenv("LOCAL_POSTGRES_URI")
MODELOS_A_PROBAR = ["qwen3.5:4b", "gemma-4:latest"]

engine = create_engine(URI_BD)

SYSTEM_PROMPT = """
Eres un agente experto en bases de datos PostgreSQL.
Dada una petición del usuario, construye la consulta SQL sintácticamente correcta,
ejecútala y responde usando los resultados obtenidos.

Reglas obligatorias:
1. Revisa SIEMPRE primero qué tablas y columnas existen con obtener_esquema antes de escribir SQL.
2. En consultas SELECT, limita los resultados a un máximo de 5 filas salvo que el usuario pida otra cantidad.
3. Si ordenas de forma descendente (DESC), añade NULLS LAST para evitar que los valores nulos salgan primero.
4. Puedes ejecutar SELECT, INSERT, UPDATE y DELETE cuando el usuario lo solicite.
5. Incluye SIEMPRE en tu respuesta final la consulta SQL exacta que ejecutaste.
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
        # Borra la línea del spinner para dejar limpio el siguiente print
        print("\r" + " " * (len(self.texto) + 5) + "\r", end="", flush=True)


# 3. HERRAMIENTAS SQL (LECTURA Y ESCRITURA)
@tool
def obtener_esquema() -> str:
    """Devuelve las tablas disponibles y las columnas con sus tipos de datos en PostgreSQL."""
    inspector = inspect(engine)
    resultado = []
    for tabla in inspector.get_table_names():
        columnas = [col["name"] + " (" + str(col["type"]) + ")" for col in inspector.get_columns(tabla)]
        resultado.append("Tabla '" + tabla + "': " + ", ".join(columnas))
    return "\n".join(resultado)


@tool
def ejecutar_sql(consulta: str) -> str:
    """Ejecuta cualquier consulta SQL (SELECT, INSERT, UPDATE, DELETE) en PostgreSQL."""
    try:
        # engine.begin() hace COMMIT automático si es INSERT, UPDATE o DELETE
        with engine.begin() as conn:
            resultado = conn.execute(text(consulta))
            if resultado.returns_rows:
                filas = resultado.fetchall()
                return str(filas) if filas else "La consulta no devolvió resultados."
            return "Operación ejecutada correctamente. Filas afectadas: " + str(resultado.rowcount)
    except Exception as e:
        return "Error SQL: " + str(e)


tools = [obtener_esquema, ejecutar_sql]


# 4. FUNCIÓN PARA PROBAR Y CRONOMETRAR CADA MODELO
def probar_modelo(nombre_modelo: str, pregunta: str):
    print("\n" + "=" * 60)
    print("PROBANDO MODELO:", nombre_modelo)
    print("=" * 60)

    llm = ChatOllama(
        model=nombre_modelo,
        temperature=0,
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
    """
    ¿Cuáles son los 3 meteoritos más pesados que cayeron después del año 1900?

    """
    pregunta_prueba = "¿Cayó algún meteorito en 2005?"

    for modelo in MODELOS_A_PROBAR:
        try:
            probar_modelo(modelo, pregunta_prueba)
        except Exception as e:
            print("\nError ejecutando", modelo + ":", e)