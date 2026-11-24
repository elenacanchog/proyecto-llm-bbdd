import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Float, Integer, Text

# 1. Leer el CSV original
df = pd.read_csv("data/Meteorite_Landings.csv")

# 2. Limpiar columnas para facilitar el trabajo al Agente SQL
df = df.drop(columns=["GeoLocation"])
df = df.rename(columns={"mass (g)": "mass_g"})
df["year"] = df["year"].astype("Int64")  # Entero que admite valores nulos (NULL)

# 3. Conectar con tu base de datos en PostgreSQL

URI_BD = os.getenv("LOCAL_POSTGRES_URI")
engine = create_engine(URI_BD)

# 4. Fijar tipos SQL explícitos para que el esquema quede impecable
tipos_sql = {
    "name": Text(),
    "id": Integer(),
    "nametype": Text(),
    "recclass": Text(),
    "mass_g": Float(),
    "fall": Text(),
    "year": Integer(),
    "reclat": Float(),
    "reclong": Float(),
}

# 5. Subir los datos a la tabla 'meteoritos'
df.to_sql(
    name="meteoritos",
    con=engine,
    if_exists="replace",
    index=False,
    dtype=tipos_sql,
)

print(f"Carga completada: {len(df)} filas insertadas en la tabla 'meteoritos'.")