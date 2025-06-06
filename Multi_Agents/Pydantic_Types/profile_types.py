# ──────────────────────────────────────────────────────────────
# profile_types.py  (v2)
# ──────────────────────────────────────────────────────────────

from typing import Union, Literal, List, Annotated
from pydantic import BaseModel, StringConstraints, model_validator, Field

# ─── Littéraux ────────────────────────────────────────────────
# === LITTÉRAUX ===
DisorderLiteral = Literal["Adhd", "AnxietyDisorders", "Depression", "OppositionalDefiantDisorder", "ConductDisorder"]
ADHDTypeLiteral = Literal["HyperactivityImpulsivityType", "InattentionType", "MixedType"]
ImpulsivitySymptomLiteral = Literal["BlurtingOut", "DifficultyWaiting", "Interrupting"]
HyperactivitySymptomLiteral = Literal["Fidgeting", "DifficultyPlayingQuietly", "ExcessiveTalking", "OnTheGo", "InabilityStaySeated", "ExcessiveRunning"]
InattentionSymptomLiteral = Literal["CarelessMistakes", "LosingThings", "EasilyDistracted", "Disorganisation", "PoorListening", "DifficultySustainingAttention", "Forgetfulness", "AvoidingTasks"]
SymptomLiteral = Union[ImpulsivitySymptomLiteral, HyperactivitySymptomLiteral, InattentionSymptomLiteral]

# ─── IDENTIFIANTS ─────────────────────────────────────────────

PatientId = Annotated[str, StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9]+$")]

DiagnosisId = Annotated[str, StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9]+Diagnosis$")]

DisorderId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Z][a-zA-Z0-9]+(Adhd|AnxietyDisorders|Depression|OppositionalDefiantDisorder|ConductDisorder)$"
    )
]

SymptomId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Z][a-zA-Z0-9]+("
        r"Forgetfulness|Interrupting|EasilyDistracted|PoorListening|"
        r"Disorganisation|DifficultySustainingAttention|BlurtingOut|"
        r"DifficultyWaiting|CarelessMistakes|LosingThings|AvoidingTasks|"
        r"Fidgeting|DifficultyPlayingQuietly|ExcessiveTalking|OnTheGo|"
        r"InabilityStaySeated|ExcessiveRunning)$"
    )
]

# ─── TRIPLETS ────────────────────────────────────────────────

class HasDiagnosisTriplet(BaseModel):
    subject: PatientId
    predicate: Literal["hasDiagnosis"]
    object: DiagnosisId


class DiagnosisDisorderTriplet(BaseModel):
    subject: DiagnosisId
    predicate: Literal["hasDiagnosisDisorder"]
    object: DisorderId


class ADHDSubtypeTriplet(BaseModel):
    subject: DisorderId  # doit correspondre à une instance Adhd
    predicate: Literal["is_A"]
    object: ADHDTypeLiteral


class SymptomTriplet(BaseModel):
    subject: DisorderId  # instance Adhd
    predicate: Literal["hasSymptom"]
    object: SymptomId


class TypingTriplet(BaseModel):
    subject: Union[DisorderId, SymptomId]
    predicate: Literal["is_A"]
    object: Union[DisorderLiteral, SymptomLiteral]


class GuidelineTriplet(BaseModel):
    subject: DiagnosisId
    predicate: Literal["hasType"]
    object: Literal["DSM-V", "DSM-IV", "CIM-11", "CIM-10"]

# ─── UNION DES TRIPLETS ──────────────────────────────────────

RDFTriplet = Union[
    HasDiagnosisTriplet,
    DiagnosisDisorderTriplet,
    ADHDSubtypeTriplet,
    SymptomTriplet,
    TypingTriplet,
    GuidelineTriplet
]

# ─── WRAPPER ─────────────────────────────────────────────────

class ExtractedTriplets_1(BaseModel):
    triplets: List[RDFTriplet] = Field(
        ..., title="Liste des triplets RDF extraits"
    )

    @model_validator(mode="after")
    def validate_unique_triplets(self) -> "ExtractedTriplets_1":
        seen = set()
        for t in self.triplets:
            key = (t.subject, t.predicate, t.object)
            if key in seen:
                raise ValueError(f"Triplet en double détecté: {key}")
            seen.add(key)
        return self