import os
from dotenv import load_dotenv
from pymongo import MongoClient

# 1. Cargamos las variables del archivo .env
load_dotenv()

# Si en tu .env la variable se llama MONGODB_URI, cámbialo aquí también
mongo_uri = os.getenv("MONGODB_URI")

def probar_conexion_airbnb():
    print("Conectando a MongoDB Atlas (sample_airbnb)...")
    
    if not mongo_uri:
        print("ERROR: No se encontró la variable MONGODB_URI en el archivo .env")
        return

    # 2. Conectamos con el cluster
    cliente = MongoClient(mongo_uri)
    
    # 3. Seleccionamos la base de datos 'sample_airbnb' y la colección 'listingsAndReviews'
    db = cliente["sample_airbnb"]
    coleccion = db["listingsAndReviews"]
    
    # 4. Contamos los alojamientos y buscamos uno en España como ejemplo
    total = coleccion.count_documents({})
    ejemplo = coleccion.find_one(
        {"address.country": "Spain"},
        {
            "name": 1,
            "property_type": 1,
            "price": 1,
            "address.market": 1,
            "host.host_name": 1,
            "amenities": 1,
            "_id": 0
        }
    )
    
    print(f"¡Conexión exitosa! Total de alojamientos disponibles: {total}")
    print("\nEjemplo de documento JSON anidado recuperado:")
    print(ejemplo)

if __name__ == "__main__":
    probar_conexion_airbnb()