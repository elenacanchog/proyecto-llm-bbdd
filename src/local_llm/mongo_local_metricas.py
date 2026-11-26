import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bson import json_util
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pymongo import MongoClient

# Carga .env desde la raíz del proyecto, independientemente del cwd.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
URI_MONGO = os.getenv("LOCAL_MONGO_URI", "mongodb://localhost:27017")
NOMBRE_BD = os.getenv("LOCAL_MONGO_DB", "sampl_airbnb")

# Edita esta lista con los modelos que tengas instalados en Ollama.
MODELOS_A_PROBAR = [
    "gemma-4:latest",
]

# Modelos omitidos: se registran como error en la gráfica sin ejecutarlos.
MODELOS_OMITIDOS = [
    "qwen3.5:4b",
]

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
1. Llama SIEMPRE primero a get_mongo_schema para ver los nombres reales de todos los campos.
2. Traduce los valores de búsqueda a como aparecen en la BD (ej. "Apartment", "Wifi", "Air conditioning").
3. Para filtrar, ordenar, limitar resultados o calcular medias/agrupaciones, usa operacion='aggregate' pasando en consulta_json un string con una LISTA JSON de etapas: '[{"$match": ...}, {"$sort": ...}, {"$limit": 5}]'.
4. No inventes datos. Si la base de datos no tiene la respuesta, di claramente que no hay información.
5. Si una consulta no devuelve resultados o hay un error de sintaxis, revisa el esquema y reintenta corrigiendo la consulta.
6. Incluye SIEMPRE en tu respuesta final la consulta o pipeline JSON exacto que ejecutaste.
"""


# =============================================================================
# 2. UTILIDADES
# =============================================================================
def resumir_valor(valor, nivel=0):
    """Compacta el esquema para no saturar el contexto del modelo."""
    if isinstance(valor, dict):
        if nivel >= 2:
            return "{...}"
        return {k: resumir_valor(v, nivel + 1) for k, v in valor.items()}
    if isinstance(valor, list):
        if not valor:
            return []
        muestra = [resumir_valor(x, nivel + 1) for x in valor[:6]]
        if len(valor) > 6:
            muestra.append(f"... (+{len(valor) - 6} elementos)")
        return muestra
    if isinstance(valor, str) and len(valor) > 40:
        return valor[:40] + "..."
    return valor


def extraer_tokens(msg) -> int:
    """Extrae tokens totales de un mensaje de LangChain/Ollama."""
    # LangChain usage_metadata (estándar actual)
    usage = getattr(msg, "usage_metadata", None)
    if usage:
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if total is not None:
                return int(total)
            return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            return int(total)
        return int(getattr(usage, "input_tokens", 0)) + int(getattr(usage, "output_tokens", 0))

    # response_metadata (compatibilidad)
    meta = getattr(msg, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if token_usage:
        total = token_usage.get("total_tokens")
        if total is not None:
            return int(total)
        return int(token_usage.get("input_tokens", 0) or token_usage.get("prompt_tokens", 0)) + int(
            token_usage.get("output_tokens", 0) or token_usage.get("completion_tokens", 0)
        )

    # Formato nativo de Ollama
    if "prompt_eval_count" in meta or "eval_count" in meta:
        return int(meta.get("prompt_eval_count", 0)) + int(meta.get("eval_count", 0))

    return 0


# =============================================================================
# 3. HERRAMIENTAS MONGODB
# =============================================================================
@tool
def get_mongo_schema() -> str:
    """Devuelve las colecciones disponibles y todos sus campos con valores de ejemplo."""
    colecciones = mongo_db.list_collection_names()
    if not colecciones:
        return "No hay colecciones en esta base de datos."

    resultado = []
    for nombre_col in colecciones:
        doc = mongo_db[nombre_col].find_one({}, CAMPOS_PESADOS)
        resumen = resumir_valor(doc) if doc else {}
        resultado.append(
            f"Colección '{nombre_col}' | Campos y ejemplo:\n"
            + json_util.dumps(resumen, indent=2, ensure_ascii=False)
        )
    return "\n\n".join(resultado)


@tool
def run_mongo(coleccion: str, operacion: str, consulta_json: Any = "{}") -> str:
    """
    Ejecuta una operación en MongoDB sobre una colección.
    - coleccion: nombre de la colección.
    - operacion: 'find', 'aggregate' o 'count'.
    - consulta_json: para 'find'/'count', un filtro JSON. Para 'aggregate', una LISTA JSON de etapas (o string con esa lista).
    """
    try:
        col = mongo_db[coleccion]
        consulta = json_util.loads(consulta_json) if isinstance(consulta_json, str) else consulta_json

        if operacion == "find":
            filas = list(col.find(consulta, CAMPOS_PESADOS).limit(5))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "No se encontraron documentos."

        if operacion == "aggregate":
            if isinstance(consulta, dict):
                consulta = [{k: v} for k, v in consulta.items()]

            tiene_project = any("$project" in etapa for etapa in consulta if isinstance(etapa, dict))
            pipeline = consulta if tiene_project else consulta + [{"$project": CAMPOS_PESADOS}]

            filas = list(col.aggregate(pipeline))
            return json_util.dumps(filas, ensure_ascii=False) if filas else "La agregación no devolvió resultados."

        if operacion == "count":
            return f"Total de documentos: {col.count_documents(consulta)}"

        return "Operación no válida. Usa únicamente: 'find', 'aggregate' o 'count'."

    except Exception as e:
        return f"Error MongoDB: {e}"


HERRAMIENTAS_MONGO = [get_mongo_schema, run_mongo]


# =============================================================================
# 4. EJECUCIÓN CON MÉTRICAS
# =============================================================================
def consultar_mongo_con_metricas(pregunta_usuario: str, modelo: str) -> dict:
    inicio_total = time.time()

    metricas = {
        "Modelo": modelo,
        "Pregunta": pregunta_usuario[:50] + ("..." if len(pregunta_usuario) > 50 else ""),
        "Exito_Consulta": False,
        "Intentos_Mongo": 0,
        "Tiempo_Total (s)": 0.0,
        "Tokens_Totales": 0,
        "Error": "Ninguno",
    }

    try:
        extra_params = {}
        nombre = modelo.lower()
        if "qwen3" in nombre or "qwen3.5" in nombre:
            # Desactiva el modo thinking en modelos Qwen para evitar bucles/lentitud.
            extra_params["reasoning"] = False

        try:
            llm = ChatOllama(
                model=modelo,
                temperature=0,
                num_ctx=8192,
                keep_alive="15m",
                num_predict=2048,
                **extra_params,
            )
        except TypeError:
            # Compatibilidad con versiones de langchain-ollama que no aceptan reasoning/num_predict.
            llm = ChatOllama(model=modelo, temperature=0, num_ctx=8192, keep_alive="15m")

        agente = create_agent(
            model=llm,
            tools=HERRAMIENTAS_MONGO,
            system_prompt=SYSTEM_PROMPT,
        )

        resultado_agente = agente.invoke(
            {"messages": [("user", pregunta_usuario)]},
            config={"recursion_limit": 25},
        )
        mensajes = resultado_agente["messages"]

        consulta_ejecutada = "No se ejecutó ninguna operación en MongoDB"
        datos_crudos = "Sin datos"
        hubo_fallo_previo = False

        for msg in mensajes:
            metricas["Tokens_Totales"] += extraer_tokens(msg)

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
                    metricas["Error"] = datos_crudos[:80]

        respuesta_final = mensajes[-1].content
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)

        print(f"\n{'=' * 80}")
        print(f"MODELO: {modelo}")
        print(f"PREGUNTA: {pregunta_usuario}")
        print(f"{'-' * 80}")
        print("[1] CONSULTA MONGODB GENERADA Y EJECUTADA:")
        print(consulta_ejecutada)
        print("\n[2] DATOS DEVUELTOS POR MONGODB:")
        print(datos_crudos)
        print("\n[3] RESPUESTA FINAL:")
        print(respuesta_final)
        print(
            f"\nMétricas: intentos={metricas['Intentos_Mongo']} | "
            f"tiempo={metricas['Tiempo_Total (s)']}s | tokens={metricas['Tokens_Totales']} | "
            f"éxito={metricas['Exito_Consulta']}"
        )

    except Exception as e:
        metricas["Error"] = f"Fallo Agente/API: {str(e)[:80]}"
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)
        print(f"\n[ERROR CRÍTICO] {modelo}: {e}")

    return metricas


# =============================================================================
# 5. GRÁFICOS COMPARATIVOS
# =============================================================================
def generar_graficos_comparativos(df: pd.DataFrame, titulo_bd: str, sufijo_archivo: str):
    ruta_raiz = Path(__file__).resolve().parents[2]
    carpeta_graficos = ruta_raiz / "graficos"
    carpeta_graficos.mkdir(exist_ok=True)

    df_plot = df.copy()
    df_plot["Num_P"] = "P" + (df_plot.groupby("Modelo").cumcount() + 1).astype(str)
    df_plot["Modelo_Corto"] = df_plot["Modelo"].str.split("/").str[-1]

    modelos = df_plot["Modelo_Corto"].unique()
    preguntas = df_plot["Num_P"].unique()
    x = range(len(preguntas))
    ancho = 0.8 / max(len(modelos), 1)
    colores = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

    # Tiempo
    fig_tiempo, ax1 = plt.subplots(figsize=(8.5, 5))
    for idx, mod in enumerate(modelos):
        sub = df_plot[df_plot["Modelo_Corto"] == mod]
        desplazamiento = [i + (idx - len(modelos) / 2 + 0.5) * ancho for i in x]
        valores = [v if ok else 0 for v, ok in zip(sub["Tiempo_Total (s)"], sub["Exito_Consulta"])]
        barras = ax1.bar(desplazamiento, valores, width=ancho, label=mod, color=colores[idx % len(colores)])
        for barra, ok, v in zip(barras, sub["Exito_Consulta"], sub["Tiempo_Total (s)"]):
            if ok:
                alto = barra.get_height()
                ax1.text(barra.get_x() + barra.get_width() / 2, alto, f"{alto:.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            else:
                ax1.text(barra.get_x() + barra.get_width() / 2, 0, "Error", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#ef4444")

    ax1.set_title(f"Tiempo de Ejecución por Consulta ({titulo_bd})", fontweight="bold")
    ax1.set_ylabel("Tiempo Total (segundos)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(preguntas)
    ax1.margins(y=0.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend()
    plt.tight_layout()
    ruta_tiempo = carpeta_graficos / f"comparativa_tiempo_{sufijo_archivo}.png"
    fig_tiempo.savefig(ruta_tiempo, dpi=300, bbox_inches="tight")

    # Tokens
    fig_tokens, ax2 = plt.subplots(figsize=(8.5, 5))
    for idx, mod in enumerate(modelos):
        sub = df_plot[df_plot["Modelo_Corto"] == mod]
        desplazamiento = [i + (idx - len(modelos) / 2 + 0.5) * ancho for i in x]
        valores = [v if ok else 0 for v, ok in zip(sub["Tokens_Totales"], sub["Exito_Consulta"])]
        barras = ax2.bar(desplazamiento, valores, width=ancho, label=mod, color=colores[idx % len(colores)])
        for barra, ok, v in zip(barras, sub["Exito_Consulta"], sub["Tokens_Totales"]):
            if ok:
                alto = barra.get_height()
                ax2.text(barra.get_x() + barra.get_width() / 2, alto, f"{int(alto):,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            else:
                ax2.text(barra.get_x() + barra.get_width() / 2, 0, "Error", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#ef4444")

    ax2.set_title(f"Consumo de Tokens por Consulta ({titulo_bd})", fontweight="bold")
    ax2.set_ylabel("Tokens Totales")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(preguntas)
    ax2.margins(y=0.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    ruta_tokens = carpeta_graficos / f"comparativa_tokens_{sufijo_archivo}.png"
    fig_tokens.savefig(ruta_tokens, dpi=300, bbox_inches="tight")

    plt.close("all")
    print(f"\nGráficos guardados en:\n - {ruta_tiempo}\n - {ruta_tokens}")


# =============================================================================
# 6. MAIN: INTRODUCE TU PREGUNTA Y GENERA GRÁFICA
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("EVALUACIÓN LOCAL MONGODB + OLLAMA")
    print("Modelos configurados:", ", ".join(MODELOS_A_PROBAR))
    print(f"Mongo URI: {URI_MONGO}")
    print(f"Base de datos: {NOMBRE_BD}")
    try:
        colecciones = mongo_db.list_collection_names()
        print(f"Colecciones visibles: {colecciones if colecciones else '(ninguna - revisa NOMBRE_BD o la conexión)'}")
    except Exception as e:
        print(f"Error conectando a MongoDB: {e}")
    print("Escribe una pregunta y pulsa Enter. Línea vacía para terminar.")
    print("=" * 80)

    preguntas = []
    while True:
        try:
            pregunta = input(f"\nPregunta {len(preguntas) + 1}: ").strip()
        except EOFError:
            break
        if not pregunta:
            break
        preguntas.append(pregunta)

    if not preguntas:
        print("No se ha introducido ninguna pregunta. Saliendo.")
        raise SystemExit(0)

    historial_metricas = []
    for modelo in MODELOS_A_PROBAR:
        print(f"\n\n{'#' * 80}")
        print(f"## MODELO: {modelo}")
        print(f"{'#' * 80}")
        for idx, pregunta in enumerate(preguntas, start=1):
            print(f"\n--- Pregunta [{idx}/{len(preguntas)}] ---")
            resultado = consultar_mongo_con_metricas(pregunta, modelo)
            historial_metricas.append(resultado)

    # Registrar modelos omitidos como error en la gráfica (sin ejecutarlos).
    for modelo in MODELOS_OMITIDOS:
        for pregunta in preguntas:
            historial_metricas.append({
                "Modelo": modelo,
                "Pregunta": pregunta[:50] + ("..." if len(pregunta) > 50 else ""),
                "Exito_Consulta": False,
                "Intentos_Mongo": 0,
                "Tiempo_Total (s)": 0.0,
                "Tokens_Totales": 0,
                "Error": "Omitido: sin respuesta / no soportado en esta prueba",
            })

    df_resultados = pd.DataFrame(historial_metricas)
    print("\n\n" + "=" * 80)
    print("RESUMEN COMPARATIVO")
    print("=" * 80)
    print(df_resultados.to_string(index=False))

    sufijo = "mongo_local"
    generar_graficos_comparativos(df_resultados, "MongoDB Local - Ollama", sufijo)
