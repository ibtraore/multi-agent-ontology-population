from typing import Literal, Union, List
from pydantic import BaseModel, Field

# === TYPES DE DÉFIS DÉFINIS PAR L’ONTOLOGIE ===
ChallengeType = Literal[
    "AcademicChallenges", "ArtChallenges", "AuditoryStimuli", "EnvironmentalStimuli", 
    "ChronicStress", "EmotionalChallenges", "ConflictWithPeers", "SocialChallenges", 
    "DifficultyMakingFriend", "ExamChallenges", "Frustration", "GroupWorkChallenges", 
    "HomeWorkChallenges", "LanguageChallenges", "LowSelfEsteem", "MathChallenges", 
    "PeerRejection", "TactileStimuli", "TeacherRelationship", "VisualStimuli"
]

# === TRIPLET RDF UTILE ===

class IsAChallengeTriplet(BaseModel):
    """Spécifie le type RDF d’un défi identifié dans le texte."""
    subject: str = Field(description="une instance du challenge (ex: 'MathematicsChallenge, ou thomas frustration').")
    predicate: Literal["is_A"] = Field(description="Relation RDF 'is_A' reliant un défi nommé à son type dans l’ontologie.")
    object: ChallengeType = Field(description="Type RDF du défi (ex: 'AcademicChallenges', 'Frustration').")

# === UNION (UNIQUE TRIPLET ICI) ===

ChallengeTriplet = Union[
    IsAChallengeTriplet
]

# === CONTENEUR FINAL ===

class ExtractedChallengeTriplets(BaseModel):
    """Conteneur des triplets RDF décrivant les défis extraits du texte."""
    triplets: List[ChallengeTriplet] = Field(
        ..., 
        description="Liste des triplets RDF valides extraits pour les défis identifiés dans le texte."
    )
