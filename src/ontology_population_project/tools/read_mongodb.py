from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from pymongo import MongoClient


def mongoCollection():
    try:
        uri = "mongodb://localhost:27017/"
        client = MongoClient(uri, timeoutMS=50000)
        database = client["patches"]
        return database["patients"]
    except Exception as e:
        raise Exception(
            "the following error occured:", e
        )

class ReadMongoDb(BaseTool):
    name: str = "Outil de lecture d'une base de données mongodb"
    description: str = (
        """ Cet outil charge le contenu d'une base de données mongodb.

            Args:
               Aucun.

            Returns:
                str: json structuré."""
    )

    def _run(self, prenom_patient= "Thomas") -> str:
        # Implementation goes here
        collection = mongoCollection()
        result = collection.find_one({"prenom" : prenom_patient},{ "_id" : 0})
            
        return result
    
# if __name__ == "__main__":
    
#     collection = mongoCollection()
#     result = collection.find_one({"prenom" : "Thomas"},{ "_id" : 0})
#     print(result)
#     # for document in result:
#     #     print(document)


