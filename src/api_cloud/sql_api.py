import os
import time
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURACIÓN DE CREDENCIALES Y CONEXIÓN A NEON (POSTGRESQL EN LA NUBE)
# =============================================================================
load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")

if not POSTGRES_URI:
    raise ValueError("ERROR: No se encontró la variable POSTGRES_URI en el archivo .env")

engine = create_engine(POSTGRES_URI)

SYSTEM_PROMPT = """
Eres un agente experto en bases de datos PostgreSQL.
Dada una petición del usuario, construye la consulta SQL sintácticamente correcta,
ejecútala y responde usando los resultados obtenidos.

Reglas obligatorias:
1. Revisa SIEMPRE primero qué tablas y columnas existen con get_schema antes de escribir SQL.
2. En consultas SELECT, limita los resultados a un máximo de 5 filas salvo que el usuario pida otra cantidad.
3. Si ordenas de forma descendente (DESC), añade NULLS LAST para evitar que los valores nulos salgan primero.
4. Puedes ejecutar SELECT, INSERT, UPDATE y DELETE cuando el usuario lo solicite.
5. Responde siempre en español de forma clara y directa basándote únicamente en los datos devueltos.
"""

# =============================================================================
# 2. HERRAMIENTAS DEL AGENTE (Nombres en inglés para evitar errores de sintaxis en LLMs)
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
# 3. UTILIDADES VISUALES PARA LA CONSOLA
# =============================================================================
def imprimir_banner_modelo(nombre_modelo: str):
    borde = "█" * 80
    print(f"\n\n{borde}")
    print(f"██  EVALUANDO MODELO (SQL API): {nombre_modelo.upper()}")
    print(f"{borde}")

def imprimir_banner_pregunta(numero: int, total: int, pregunta: str):
    linea = "═" * 80
    print(f"\n{linea}")
    print(f"▶ PREGUNTA [{numero}/{total}]: {pregunta}")
    print(f"{linea}")

# =============================================================================
# 4. MOTOR PRINCIPAL DEL AGENTE TEXT-TO-SQL CON TELEMETRÍA
# =============================================================================
def consultar_meteoritos_con_metricas(pregunta_usuario: str, modelo: str) -> dict:
    inicio_total = time.time()

    metricas = {
        "Modelo": modelo,
        "Pregunta": pregunta_usuario[:40] + "...",
        "Exito_Consulta": False,
        "Intentos_SQL": 0,
        "Tiempo_Total (s)": 0.0,
        "Tokens_Totales": 0,
        "Error": "Ninguno"
    }

    try:
        llm = ChatGroq(model=modelo, temperature=0)

        agente = create_agent(
            model=llm,
            tools=HERRAMIENTAS_SQL,
            system_prompt=SYSTEM_PROMPT,
        )

        resultado_agente = agente.invoke({"messages": [("user", pregunta_usuario)]})
        mensajes = resultado_agente["messages"]

        sql_ejecutado = "No se generó ninguna consulta SQL"
        datos_crudos = "Sin datos"
        hubo_fallo_previo = False

        for msg in mensajes:
            if hasattr(msg, "response_metadata") and msg.response_metadata:
                uso_tokens = msg.response_metadata.get("token_usage", {})
                metricas["Tokens_Totales"] += uso_tokens.get("total_tokens", 0)

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
                    metricas["Error"] = datos_crudos[:50]

        respuesta_final = mensajes[-1].content
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)

        print("\n[1] CONSULTA SQL GENERADA Y EJECUTADA POR EL AGENTE:")
        print(sql_ejecutado)

        print("\n[2] DATOS DEVUELTOS POR POSTGRESQL (NEON):")
        print(datos_crudos)

        print("\n[3] RESPUESTA FINAL:")
        print(respuesta_final)
        print(f"\n⏱️  Métricas: Intentos SQL: {metricas['Intentos_SQL']} | Tiempo Total: {metricas['Tiempo_Total (s)']}s | Tokens: {metricas['Tokens_Totales']}")

    except Exception as e:
        metricas["Error"] = f"Fallo Agente/API: {str(e)[:50]}"
        metricas["Tiempo_Total (s)"] = round(time.time() - inicio_total, 2)
        print(f"\n[ERROR CRÍTICO]: {e}")

    return metricas

# =============================================================================
# 5. FUNCIÓN PARA GENERAR Y GUARDAR GRÁFICOS COMPARATIVOS
# =============================================================================

def generar_graficos_comparativos(df: pd.DataFrame, titulo_bd: str, sufijo_archivo: str):
    """Genera, guarda por separado y muestra en ventana los gráficos de tiempo y tokens."""
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
    colores = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]

    # -------------------------------------------------------------------------
    # GRÁFICO 1: TIEMPO DE EJECUCIÓN (SEGUNDOS)
    # -------------------------------------------------------------------------
    fig_tiempo, ax1 = plt.subplots(figsize=(8.5, 5))
    for idx, mod in enumerate(modelos):
        sub = df_plot[df_plot["Modelo_Corto"] == mod]
        desplazamiento = [i + (idx - len(modelos) / 2 + 0.5) * ancho for i in x]
        barras = ax1.bar(desplazamiento, sub["Tiempo_Total (s)"], width=ancho, label=mod, color=colores[idx % len(colores)])
        for barra, ok in zip(barras, sub["Exito_Consulta"]):
            alto = barra.get_height()
            etiqueta = f"{alto:.1f}s" if ok else "Error"
            ax1.text(barra.get_x() + barra.get_width() / 2, alto, etiqueta,
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

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

    # -------------------------------------------------------------------------
    # GRÁFICO 2: CONSUMO DE TOKENS
    # -------------------------------------------------------------------------
    fig_tokens, ax2 = plt.subplots(figsize=(8.5, 5))
    for idx, mod in enumerate(modelos):
        sub = df_plot[df_plot["Modelo_Corto"] == mod]
        desplazamiento = [i + (idx - len(modelos) / 2 + 0.5) * ancho for i in x]
        barras = ax2.bar(desplazamiento, sub["Tokens_Totales"], width=ancho, label=mod, color=colores[idx % len(colores)])
        for barra, ok in zip(barras, sub["Exito_Consulta"]):
            alto = barra.get_height()
            etiqueta = f"{int(alto):,}" if ok else "0"
            ax2.text(barra.get_x() + barra.get_width() / 2, alto, etiqueta,
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

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

    print(f"\n📊 Gráficos guardados en:\n   - {ruta_tiempo}\n   - {ruta_tokens}")

    # Abre ambos gráficos en ventanas interactivas al finalizar
    plt.show()
    plt.close("all")

# =============================================================================
# 6. BATERÍA DE EVALUACIÓN COMPARATIVA
# =============================================================================
if __name__ == "__main__":
    MIS_MODELOS = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "allam-2-7b"
    ]

    MIS_PREGUNTAS = [
        "¿Cayó algún meteorito en el año 2005? Dime 3 ejemplos.",
        "¿Cuáles son los 3 meteoritos más pesados (mass_g) que cayeron después del año 1900?",
        "Dime las 5 clases de meteoritos (recclass) más frecuentes y cuál es su peso medio."
    ]

    historial_metricas = []

    for modelo in MIS_MODELOS:
        imprimir_banner_modelo(modelo)
        
        for idx, pregunta in enumerate(MIS_PREGUNTAS, start=1):
            imprimir_banner_pregunta(idx, len(MIS_PREGUNTAS), pregunta)
            res = consultar_meteoritos_con_metricas(pregunta, modelo)
            historial_metricas.append(res)
            time.sleep(2)

    print("\n\n" + "★" * 80)
    print("RESUMEN COMPARATIVO DE RENDIMIENTO (POSTGRESQL NEON + API)")
    print("★" * 80)
    df_resultados = pd.DataFrame(historial_metricas)
    print(df_resultados.to_string(index=False))

    generar_graficos_comparativos(df_resultados, "PostgreSQL Neon - API", "sql_api")