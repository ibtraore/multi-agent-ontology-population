from typing import Literal, Union, List, Annotated
from pydantic import BaseModel, StringConstraints, Field, model_validator

# ─── Contraintes de nommage pour les identifiants ──────────────────────────────

SituationId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9]+Situation$"),
    Field(description="Identifiant unique d'une situation, au format '<Prénom>Situation' (ex: 'ThomasSituation')")
]

PersonId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9]+$"),
    Field(description="Identifiant d'une personne (enfant ou adulte), format sans espaces, ex: 'Thomas', 'MmeMartin'")
]

ActivityId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9]+$"),
    Field(description="Nom d’une activité encodée, ex: 'MathLesson', 'FreePlayActivity'")
]

IntervalId = Annotated[
    str,
    StringConstraints(pattern=r"^SituationInterval$"),
    Field(description="Nom fixe de l’intervalle temporel associé à une situation : 'SituationInterval'")
]

EnvironmentId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9]+$"),
    Field(description="Nom de l’environnement associé à une situation, ex: 'SchoolClassroom', 'Playground'")
]

ChallengeId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9]+$"),
    Field(description="Nom d’un défi ou problème rencontré, ex: 'MathChallenge'")
]

# ─── Catégories de situations (ontologie) ─────────────────────────────────────

SituationConcept = Literal[
    "AcademicSituation", "EmotionalRegulationSituation",
    "FamilyDynamicSituation", "OrganizationSituation",
    "SocialSituation", "TimeManagementSituation"
]

# ─── Modèles de triplets RDF ─────────────────────────────────────────────────

class Is_ATriplet(BaseModel):
    """Définit le type ontologique d'une instance de situation."""
    subject: SituationId
    predicate: Literal["is_A"]
    object: SituationConcept = Field(description="Catégorie de la situation selon l’ontologie médicale.")

class HasPatientTriplet(BaseModel):
    """Relie une situation à l’enfant concerné."""
    subject: SituationId
    predicate: Literal["hasPatient"]
    object: PersonId = Field(description="Nom du patient (enfant) concerné par cette situation.")

class HasTemporalEntityTriplet(BaseModel):
    """Relie une situation à son intervalle temporel."""
    subject: SituationId
    predicate: Literal["hasTemporalEntity"]
    object: IntervalId = Field(description="Instance temporelle nommée 'SituationInterval' associée à la situation.")

class HasEnvironmentTriplet(BaseModel):
    """Relie une situation à son environnement contextuel."""
    subject: SituationId
    predicate: Literal["hasEnvironment"]
    object: EnvironmentId = Field(description="Nom de l’environnement (salle de classe, cour, etc.).")

class HasChallengeTriplet(BaseModel):
    """Indique les challenges rencontrés dans la situation décrite."""
    subject: SituationId
    predicate: Literal["hasChallenge"]
    object: ChallengeId = Field(description="Nom du défi associé à la situation (ex: 'NoiseChallenge').")

class IsEngagedInTriplet(BaseModel):
    """Relie un enfant à l’activité dans laquelle il est engagé."""
    subject: PersonId
    predicate: Literal["isEngagedIn"]
    object: ActivityId = Field(description="Nom de l’activité en cours, liée à la situation.")

class IsWithTriplet(BaseModel):
    """Relie une activité à la personne (enseignant/animateur) qui y participe avec l’enfant."""
    subject: ActivityId
    predicate: Literal["isWith"]
    object: PersonId = Field(description="Nom de la personne encadrante présente durant l’activité.")

# ─── Union de tous les types de triplets intermodulaires ─────────────────────

InterModuleTriplet = Union[
    Is_ATriplet,
    HasPatientTriplet,
    HasTemporalEntityTriplet,
    HasEnvironmentTriplet,
    HasChallengeTriplet,
    IsEngagedInTriplet,
    IsWithTriplet
]

# ─── Modèle principal regroupant les triplets du module Situation ────────────

class ExtractedSituationTriplets(BaseModel):
    """
    Conteneur de tous les triplets RDF intermodulaires représentant une situation.
    Ces triplets décrivent les liens entre la situation et les autres modules (personne, activité, environnement, temps, challenge).
    """
    triplets: List[InterModuleTriplet] = Field(
        ..., description="Liste de triplets RDF extraits, formalisant les relations entre modules pour une situation donnée."
    )

    @model_validator(mode="after")
    def check_no_duplicates(self) -> "ExtractedSituationTriplets":
        """Vérifie qu’aucun triplet n’est dupliqué dans la sortie."""
        seen = set()
        for triplet in self.triplets:
            key = (triplet.subject, triplet.predicate, triplet.object)
            if key in seen:
                raise ValueError(f"Duplicate triplet detected: {key}")
            seen.add(key)
        return self
