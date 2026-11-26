import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from sqlalchemy import create_engine, inspect, text

load_dotenv()

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
URI_BD = os.getenv("LOCAL_POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/nasa_meteoritos")

# Edita esta lista con los modelos que tengas instalados en Ollama.
MODELOS_A_PROBAR = [
    "qwen3.5:4b",
    "gemma-4:latest",
]

engine = create_engine(URI_BD)

SYSTEM_PROMPT = """
Eres un agente experto en bases de datos PostgreSQL.
Dada una petición del usuario, construye la consulta SQL sintácticamente correcta,
ejecútala y responde usando los resultados obtenidos.

Reglas obligatorias:
1. Revisa SIEMPRE primero qué tablas y columnas existen con get_schema antes de escribir SQL.
2. En consultas SELECT, limita los resultados a un máximo de 5 filas salvo que el usuario pida otra cantidad.
3. Si ordenas de forma descendente (DESC), añade NULLS LAST para evitar que los valores nulos salgan primero.
4. Puedes ejecutar SELECT, INSERT, UPDATE y DELETE cuando el usuario lo solicite.
5. Si una consulta falla, revisa el error, corrige la SQL y reintenta.
6. Incluye SIEMPRE en tu respuesta final la consulta SQL exacta que ejecutaste.
"""


# =============================================================================
# 2. UTILIDADES
# =============================================================================
def extraer_tokens(msg) -> int:
    """Extrae tokens totales de un mensaje de LangChain/Ollama."""
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

    meta = getattr(msg, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if token_usage:
        total = token_usage.get("total_tokens")
        if total is not None:
            return int(total)
        return int(token_usage.get("input_tokens", 0) or token_usage.get("prompt_tokens", 0)) + int(
            token_usage.get("output_tokens", 0) or token_usage.get("completion_tokens", 0)
        )

    if "prompt_eval_count" in meta or "eval_count" in meta:
        return int(meta.get("prompt_eval_count", 0)) + int(meta.get("eval_count", 0))

    return 0


# =============================================================================
# 3. HERRAMIENTAS SQL
# =============================================================================
@tool
def get_schema() -> str:
    """Devuelve las tablas disponibles y las columnas con sus tipos de datos en PostgreSQL."""
    inspector = inspect(engine)
    resultado = []
    for tabla in inspector.get_table_names():
        columnas = [f"{col['name']} ({col['type']})" for col in inspector.get_columns(tabla)]
        resultado.append(f"Tabla '{tabla}': {', '.join(columnas)}")
    return "\n".join(resultado)


@tool
def run_sql(consulta: str) -> str:
    """Ejecuta cualquier consulta SQL (SELECT, INSERT, UPDATE, DELETE) en PostgreSQL."""
    try:
        with engine.begin() as conn:
            resultado = conn.execute(text(consulta))
            if resultado.returns_rows:
                filas = resultado.fetchall()
                return str(filas) if filas else "La consulta no devolvió resultados."
            return f"Operación ejecutada correctamente. Filas afectadas: {resultado.rowcount}"
    except Exception as e:
        return f"Error SQL: {e}"


HERRAMIENTAS_SQL = [get_schema, run_sql]


# =============================================================================
# 4. EJECUCIÓN CON MÉTRICAS
# =============================================================================
def consultar_sql_con_metricas(pregunta_usuario: str, modelo: str) -> dict:
    inicio_total = time.time()

    metricas = {
        "Modelo": modelo,
        "Pregunta": pregunta_usuario[:50] + ("..." if len(pregunta_usuario) > 50 else ""),
        "Exito_Consulta": False,
        "Intentos_SQL": 0,
        "Tiempo_Total (s)": 0.0,
        "Tokens_Totales": 0,
        "Error": "Ninguno",
    }

    try:
        llm = ChatOllama(model=modelo, temperature=0, num_ctx=8192)

        agente = create_agent(
            model=llm,
            tools=HERRAMIENTAS_SQL,
            system_prompt=SYSTEM_PROMPT,
        )

        resultado_agente = agente.invoke(
            {"messages": [("user", pregunta_usuario)]},
            config={"recursion_limit": 14},
        )
        mensajes = resultado_agente["messages"]

        sql_ejecutado = "No se generó ninguna consulta SQL"
        datos_crudos = "Sin datos"
        hubo_fallo_previo = False

        for msg in mensajes:
            metricas["Tokens_Totales"] += extraer_tokens(msg)

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for llamada in msg.tool_calls:
                    if llamada["name"] == "run_sql":
                        metricas["Intentos_SQL"] += 1
                        sql_ejecutado = llamada["args"].get("consulta", "")

            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "run_sql":
                datos_crudos = msg.content
                if not datos_crudos.startswith("Error SQL:"):
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
        print("[1] CONSULTA SQL GENERADA Y EJECUTADA:")
        print(sql_ejecutado)
        print("\n[2] DATOS DEVUELTOS POR POSTGRESQL:")
        print(datos_crudos)
        print("\n[3] RESPUESTA FINAL:")
        print(respuesta_final)
        print(
            f"\nMétricas: intentos={metricas['Intentos_SQL']} | "
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

    fig_tiempo, ax1 = plt.subplots(figsize=(8.5, 5))
    for idx, mod in enumerate(modelos):
        sub = df_plot[df_plot["Modelo_Corto"] == mod]
        desplazamiento = [i + (idx - len(modelos) / 2 + 0.5) * ancho for i in x]
        barras = ax1.bar(desplazamiento, sub["Tiempo_Total (s)"], width=ancho, label=mod, color=colores[idx % len(colores)])
        for barra, ok in zip(barras, sub["Exito_Consulta"]):
            alto = barra.get_height()
            etiqueta = f"{alto:.1f}s" if ok else "Error"
            ax1.text(barra.get_x() + barra.get_width() / 2, alto, etiqueta, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

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

    fig_tokens, ax2 = plt.subplots(figsize=(8.5, 5))
    for idx, mod in enumerate(modelos):
        sub = df_plot[df_plot["Modelo_Corto"] == mod]
        desplazamiento = [i + (idx - len(modelos) / 2 + 0.5) * ancho for i in x]
        barras = ax2.bar(desplazamiento, sub["Tokens_Totales"], width=ancho, label=mod, color=colores[idx % len(colores)])
        for barra, ok in zip(barras, sub["Exito_Consulta"]):
            alto = barra.get_height()
            etiqueta = f"{int(alto):,}" if ok else "0"
            ax2.text(barra.get_x() + barra.get_width() / 2, alto, etiqueta, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

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
    print("EVALUACIÓN LOCAL SQL + OLLAMA")
    print("Modelos configurados:", ", ".join(MODELOS_A_PROBAR))
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
            resultado = consultar_sql_con_metricas(pregunta, modelo)
            historial_metricas.append(resultado)

    df_resultados = pd.DataFrame(historial_metricas)
    print("\n\n" + "=" * 80)
    print("RESUMEN COMPARATIVO")
    print("=" * 80)
    print(df_resultados.to_string(index=False))

    generar_graficos_comparativos(df_resultados, "PostgreSQL Local - Ollama", "sql_local")
