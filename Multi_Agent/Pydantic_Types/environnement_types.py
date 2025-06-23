from typing import Literal, Union, List
from pydantic import BaseModel, Field


# === TYPES D’ENVIRONNEMENT DÉFINIS PAR L’ONTOLOGIE ===
EnvironmentType = Literal[
    "Community", "LivingEnvironment", "FamilyHome", "Garden", "PlayingEnvironment", 
    "HomePlaySpace", "HomeSchooling", "LearningEnvironment", "Library", "OnlineLearning",
    "Park", "PreSchool", "PublicPlayground", "School", "SchoolResidence", 
    "SportComplexe", "SummerCamp", "Theater"
]

class IsAEnvironmentTriplet(BaseModel):
    """Spécifie le type RDF d’un environnement (ex: 'SchoolClassroom' est un 'LearningEnvironment')."""
    subject: str = Field(description="Nom ou identifiant de l’environnement extrait du texte (ex: 'SchoolClassroom').")
    predicate: Literal["is_A"] = Field(description="Relation RDF 'is_A' entre une instance d’environnement et son type ontologique.")
    object: EnvironmentType = Field(description="Type RDF de l’environnement (ex: 'Library', 'Park', etc.).")

# === UNION DES TRIPLETS ENVIRONNEMENTAUX ===

EnvironmentTriplet = Union[
    IsAEnvironmentTriplet
]

# === CONTENEUR FINAL ===

class ExtractedEnvironmentTriplets(BaseModel):
    """Conteneur contenant les triplets RDF représentant les environnements extraits du texte."""
    triplets: List[EnvironmentTriplet] = Field(
        ..., 
        description="Liste des triplets RDF valides extraits concernant les environnements du texte."
    )
