from typing import Literal, Union, List
from pydantic import BaseModel, Field

# === TYPES D’ACTIVITÉS DÉFINIS DANS L’ONTOLOGIE ===
ActivityType = Literal[
    "Adventure", "OutdoorActivity", "ArtCourse", "EducationalActivity", "BoardGame",
    "GameActivity", "CommunityService", "SocialActivity", "Conversation", "FamillyActivity",
    "Cooking", "CulturalActivity", "CulturalFestival", "CulturalLearning", "Eating",
    "EnvironmentalProject", "Fitness", "PhysicalActivity", "GroupGame", "GroupWork",
    "HomeWork", "LanguageCourse", "MathCourse", "MuseumVisit", "NatureExploration",
    "Picnic", "Puzzle", "SchoolExam", "Sport", "VideoGame", "WatchingMovie"
]

class IsAActivityTriplet(BaseModel):
    """Relation RDF 'is_A' indiquant le type d’une activité identifiée dans le texte."""
    subject: str = Field(description="Nom de l’activité mentionnée dans le texte (ex: 'SocialActivity').")
    predicate: Literal["is_A"] = Field(description="Relation RDF 'is_A' reliant une activité à son type RDF.")
    object: ActivityType = Field(description="Type RDF de l’activité (ex: 'EducationalActivity').")

# === UNION DES TRIPLETS DU MODULE ACTIVITÉ ===
ActivityTriplet = Union[
    IsAActivityTriplet
]

class ExtractedActivityTriplets(BaseModel):
    """Conteneur des triplets RDF d’activités extraits du texte."""
    triplets: List[ActivityTriplet] = Field(
        ..., 
        description="Liste des triplets RDF valides représentant les types d’activités mentionnées dans le texte."
    )
