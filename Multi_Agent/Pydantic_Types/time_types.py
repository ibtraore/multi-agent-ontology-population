from typing import Literal, Union, List
from pydantic import BaseModel, Field



# === CLASSES DE BASE POUR CHAQUE RELATION TEMPORELLE ===
class Is_ATriplet(BaseModel):
    """Relation spécifiant une instance de concept."""
    subject: str = Field(description="Nom de l’instance de type Instant ou Interval (ex: 'StartInstant_1', 'SituationInterval_2'...).")
    predicate: Literal["is_A"] = Field(description="Relation indiquant une instance de concept.")
    object: Literal["Interval", "Instant"] = Field (...,description=  "c'est un concept de l'ontologie" )

class HasDayTriplet(BaseModel):
    """Relation spécifiant le jour d’un instant donné."""
    subject: str = Field(description="Nom de l’instance de type Instant (ex: 'StartInstant', 'BirthInstant').")
    predicate: Literal["hasDay"] = Field(description="Relation indiquant le jour du mois pour un instant donné.")
    object: int = Field(ge=1, le=31, description="Jour du mois, valeur entière entre 1 et 31.")

class HasHourTriplet(BaseModel):
    """Relation spécifiant l’heure d’un instant donné."""
    subject: str = Field(description="Nom de l’instance de type Instant (ex: 'StartInstant').")
    predicate: Literal["hasHour"] = Field(description="Relation indiquant l’heure (en format 24h) pour un instant donné.")
    object: int = Field(ge=0, le=23, description="Heure (entre 0 et 23).")

class HasMinuteTriplet(BaseModel):
    """Relation spécifiant les minutes d’un instant donné."""
    subject: str = Field(description="Nom de l’instance de type Instant.")
    predicate: Literal["hasMinute"] = Field(description="Relation indiquant les minutes pour un instant donné.")
    object: int = Field(ge=0, le=59, description="Minute de l’heure (entre 0 et 59).")

class HasMonthTriplet(BaseModel):
    """Relation spécifiant le mois d’un instant donné."""
    subject: str = Field(description="Nom de l’instance de type Instant.")
    predicate: Literal["hasMonth"] = Field(description="Relation indiquant le mois pour un instant donné.")
    object: int = Field(ge=1, le=12, description="Mois de l’année (entre 1 et 12).")

class HasSecondTriplet(BaseModel):
    """Relation spécifiant les secondes d’un instant donné."""
    subject: str = Field(description="Nom de l’instance de type Instant.")
    predicate: Literal["hasSecond"] = Field(description="Relation indiquant les secondes pour un instant donné.")
    object: int = Field(ge=0, le=59, description="Seconde de la minute (entre 0 et 59).")

class HasYearTriplet(BaseModel):
    """Relation spécifiant l’année d’un instant donné."""
    subject: str = Field(description="Nom de l’instance de type Instant.")
    predicate: Literal["hasYear"] = Field(description="Relation indiquant l’année pour un instant donné.")
    object: int = Field(description="Année au format AAAA, par exemple 2024.")

class HasEndTriplet(BaseModel):
    """Relation liant un intervalle temporel à son instant de fin."""
    subject: str = Field(description="Nom de l’instance de type Interval (ex: 'SituationInterval').")
    predicate: Literal["hasEnd"] = Field(description="Relation indiquant l’instant de fin d’un intervalle.")
    object: str = Field(description="Nom de l’instance de type Instant représentant la fin (ex: 'EndInstant').")

class HasStartTriplet(BaseModel):
    """Relation liant un intervalle temporel à son instant de début."""
    subject: str = Field(description="Nom de l’instance de type Interval (ex: 'SituationInterval').")
    predicate: Literal["hasStart"] = Field(description="Relation indiquant l’instant de début d’un intervalle.")
    object: str = Field(description="Nom de l’instance de type Instant représentant le début (ex: 'StartInstant').")

# === UNION DE TOUS LES TRIPLETS TEMPORELS ===

TimeTriplet = Union[
    Is_ATriplet,
    HasDayTriplet,
    HasHourTriplet,
    HasMinuteTriplet,
    HasMonthTriplet,
    HasSecondTriplet,
    HasYearTriplet,
    HasEndTriplet,
    HasStartTriplet,
]

# === CONTENEUR FINAL ===

class ExtractedTimeTriplets(BaseModel):
    """Conteneur contenant l’ensemble des triplets temporels extraits du texte."""
    triplets: List[TimeTriplet] = Field(
        ...,
        description="Liste structurée des relations temporelles extraites du document, au format RDF compatible."
    )
