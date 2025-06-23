from typing import Literal, Union, List
from pydantic import BaseModel, Field
from datetime import date

# === CONCEPTS DE L'ONTOLOGIE (strictement ceux de la spécification) ===
PersonConcept = Literal[
    "Person","Patient", "Doctor",
    "Child", "Father", "Parent", "Mother", "Teacher"
]
# === MODÈLES DE TRIPLETS PERSONNELS (relations RDF explicites) ===
class Is_ATriplet(BaseModel):
    """Relation spécifiant une instance de concept."""
    subject: str = Field(description="Nom de l’instance de type Person (ex: nom de patient ou médecin ou ...)")
    predicate: Literal["is_A"] = Field(description="Relation indiquant une instance de concept.")
    object: PersonConcept= Field (...,description= "c'est un concept de l'ontologie prédinie telques : /" 
                                    "Person,Patient, Doctor,Child, Father, Parent, Mother, Teacher" )

class HasAgeTriplet(BaseModel):
    """Relation 'hasAge' associant un patient à son âge."""
    subject: str = Field(description="Identifiant du patient concerné (ex : 'Thomas').")
    predicate: Literal["hasAge"] = Field(description="Relation indiquant l’âge d’un patient.")
    object: int = Field(ge=0, description="Âge du patient en années (ex : 6).")

class HasDateOfBirthTriplet(BaseModel):
    subject: str = Field(description="Identifiant du patient (ex : 'Thomas').")
    predicate: Literal["hasDateOfBirth"] = Field(description="Relation indiquant la date de naissance d’un patient.")
    object: str = Field(description="Date de naissance au format ISO (ex : '2018-03-30').")


class HasGenderTriplet(BaseModel):
    """Relation 'hasGender' applicable à toute personne."""
    subject: str = Field(description="Identifiant de la personne concernée (Patient, Parent, Doctor...), ex : 'Thomas'.")
    predicate: Literal["hasGender"] = Field(description="Relation indiquant le genre d’une personne.")
    object: str = Field(description="Genre tel qu’exprimé dans le texte source (ex : 'Male', 'Féminin').")

class HasNameTriplet(BaseModel):
    """Relation 'hasName' applicable à toute personne."""
    subject: str = Field(description="Identifiant de la personne (ex : 'Sylvie', 'Thomas').")
    predicate: Literal["hasName"] = Field(description="Relation associant une personne à son nom.")
    object: str = Field(description="Nom tel qu’il apparaît dans le texte (ex : 'Thomas', 'Dr. Dupont').")

class HasServiceTriplet(BaseModel):
    """Relation 'hasService' applicable à un docteur, indiquant son service médical."""
    subject: str = Field(description="Identifiant du docteur concerné (ex : 'DrMartin').")
    predicate: Literal["hasService"] = Field(description="Relation précisant le service médical d’un docteur.")
    object: str = Field(description="Nom du service médical (ex : 'Pédiatrie', 'Psychiatrie').")



# === UNION DE TOUS LES TRIPLETS ===

PersonTriplet = Union[
    Is_ATriplet,
    HasAgeTriplet,
    HasDateOfBirthTriplet,
    HasGenderTriplet,
    HasNameTriplet,
    HasServiceTriplet,
]

# === CONTENEUR FINAL ===

class ExtractedPersonTriplets(BaseModel):
    """Conteneur regroupant les triplets RDF concernant une personne, patient ou entourage."""
    triplets: List[PersonTriplet] = Field(
        ..., 
        description="Liste des triplets RDF valides extraits du texte concernant les personnes."
    )
