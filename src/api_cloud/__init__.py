import os
from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
cliente = MongoClient(os.getenv("MONGODB_URI"))
coleccion = cliente["sample_airbnb"]["listingsAndReviews"]

print("Descargando documentos de Atlas...")
documentos = list(coleccion.find({}))

with open("data/sample_airbnb.json", "w", encoding="utf-8") as archivo:
    archivo.write(json_util.dumps(documentos, ensure_ascii=False, indent=2))

print(f"¡Hecho! Se han guardado {len(documentos)} documentos en data/sample_airbnb.json")