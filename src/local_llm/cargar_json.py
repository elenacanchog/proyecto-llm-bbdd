import os
from pymongo import MongoClient
from bson import json_util

# 1. Conectar al servidor local de MongoDB
URI_MONGO = os.getenv("LOCAL_MONGO_URI")
cliente = MongoClient(URI_MONGO)
db = cliente["sampl_airbnb"]       # Crea la BD automáticamente al insertar datos
coleccion = db["datos"]           # Nombre de tu colección

# 2. Leer el archivo JSON (json_util soporta JSON normal y formatos propios de Mongo como $oid)
with open("sample_airbnb.json", "r", encoding="utf-8") as archivo:
    contenido = archivo.read().strip()

try:
    datos = json_util.loads(contenido)
except Exception:
    # Por si el archivo tiene un objeto JSON por línea (formato NDJSON)
    datos = [json_util.loads(linea) for linea in contenido.splitlines() if linea.strip()]

# 3. Limpiar si ya existía e insertar
coleccion.delete_many({})
if isinstance(datos, list):
    coleccion.insert_many(datos)
else:
    coleccion.insert_one(datos)

print("Carga completada. Documentos insertados:", coleccion.count_documents({}))