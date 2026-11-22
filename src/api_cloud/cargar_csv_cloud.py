import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.types import Float, Integer, Text

# 1. Cargamos la URL secreta de Neon desde el archivo .env
load_dotenv()
URI_BD = os.getenv("POSTGRES_URI")

if not URI_BD:
    raise ValueError("ERROR: No se encontró la variable POSTGRES_URI en el archivo .env")

# 2. Leer el CSV original desde la carpeta data/
print("Leyendo el archivo Meteorite_Landings.csv...")
df = pd.read_csv("data/Meteorite_Landings.csv")

# 3. Limpiar columnas (exactamente igual que en la versión local para mantener compatibilidad)
df = df.drop(columns=["GeoLocation"])
df = df.rename(columns={"mass (g)": "mass_g"})
df["year"] = df["year"].astype("Int64")  # Entero que admite valores nulos (NULL)

# 4. Conectar con tu base de datos PostgreSQL en la nube (Neon)
print("Conectando con PostgreSQL en Neon...")
engine = create_engine(URI_BD)

# 5. Fijar tipos SQL explícitos para que el esquema de la tabla sea limpio
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

# 6. Subir los datos a la tabla 'meteoritos'
print("Subiendo registros a la nube (puede tardar unos 10-20 segundos)...")
df.to_sql(
    name="meteoritos",
    con=engine,
    if_exists="replace",
    index=False,
    dtype=tipos_sql,
)

print(f"¡Carga completada con éxito! {len(df)} filas insertadas en la tabla 'meteoritos'.")