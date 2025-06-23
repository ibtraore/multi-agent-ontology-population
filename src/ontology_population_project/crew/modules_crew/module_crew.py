from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os 
from dotenv import load_dotenv
from src.ontology_population_project.pydantic_types.person_types import ExtractedPersonTriplets
from src.ontology_population_project.pydantic_types.time_types import ExtractedTimeTriplets
from src.ontology_population_project.pydantic_types.environnement_types import ExtractedEnvironmentTriplets
from src.ontology_population_project.pydantic_types.challenge_types import ExtractedChallengeTriplets
from src.ontology_population_project.pydantic_types.activity_types import ExtractedActivityTriplets
from src.ontology_population_project.pydantic_types.situation_types import ExtractedSituationTriplets
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

load_dotenv()
os.environ["OTEL_SDK_DISABLED"] = "true"

# Base directory configuration (will be modified dynamically)
BASE_GRADIO_OUTPUT_DIR = "src/ontology_population_project/agent-output/Gradio"
os.makedirs(BASE_GRADIO_OUTPUT_DIR, exist_ok=True)

# Global variable for output directory (will be set dynamically)
current_module_output_dir = None

def set_module_output_directory(patient_name: str) -> str:
    """Configure output directory for a given patient"""
    global current_module_output_dir
    current_module_output_dir = os.path.join(BASE_GRADIO_OUTPUT_DIR, patient_name)
    os.makedirs(current_module_output_dir, exist_ok=True)
    return current_module_output_dir

def get_current_module_output_dir() -> str:
    """Return the currently configured output directory"""
    global current_module_output_dir
    if current_module_output_dir is None:
        # Default directory if no patient is defined
        current_module_output_dir = os.path.join(BASE_GRADIO_OUTPUT_DIR, "default_patient")
        os.makedirs(current_module_output_dir, exist_ok=True)
    return current_module_output_dir

# LLM Configuration
model_id = "mistral/mistral-large-latest"

# Advanced configuration with detailed parameters
llm = LLM(
    model=model_id, 
    temperature=0,          # Higher for more creative outputs
    timeout=120,            # Seconds to wait for response
    seed=42                 # For reproducible results
)

# Project context definition
about_project = """
Le projet vise à extraire automatiquement des graphes RDF à partir de textes libres décrivant des scènes de vie d'enfants,
 en s'appuyant sur une architecture à ontologies modulaires.

 L'objectif est de transformer chaque description textuelle en un ensemble de triplets RDF valides et cohérents,
 organisés autour de six modules : Temps, Personne, Environnement, Activité, Challenge et un module central Situation chargé de relier les entités.

  Le système repose sur un pipeline mono_agent et multi_task :

  - `extracting_module_agent` : exécute six tâches d'extraction RDF en parallèle, une par module.

  Les triplets doivent être :
  - syntaxiquement valides (structure Pydantic),
  - justifiés explicitement par le texte (pas d'invention),
  - conformes aux relations définies dans l'ontologie.

  Les modules sdoivent etre liés au module Situation pour construire une représentation complète, connectée et exploitable par raisonnement sémantique.
"""

project_context = StringKnowledgeSource(
    content=about_project
)

@CrewBase
class ModuleCrew():
    """Ontology Module Extraction Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    def __init__(self, patient_name: str = None):
        """
        Initialize ModuleCrew with a patient name to define dynamic output directory
        
        Args:
            patient_name (str): Patient name to create dynamic output directory
        """
        super().__init__()
        if patient_name:
            self.output_dir = set_module_output_directory(patient_name)
        else:
            self.output_dir = get_current_module_output_dir()

    def set_patient_output_dir(self, patient_name: str):
        """
        Update output directory for a new patient
        
        Args:
            patient_name (str): Patient name
        """
        self.output_dir = set_module_output_directory(patient_name)

    @agent
    def extracting_module_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['extracting_module_agent'], # type: ignore[index]
            llm= llm,
            verbose=True,
        )

    
    @task
    def situation_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['situation_extraction_task'], # type: ignore[index]
            output_json= ExtractedSituationTriplets,
            output_file=os.path.join(self.output_dir, "module_extraction_situation.json")
        )
    
    @task
    def time_triplet_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['time_triplet_extraction_task'], # type: ignore[index]
            context= [self.situation_extraction_task()],  # Depends on situation task
            output_json= ExtractedTimeTriplets,
            output_file=os.path.join(self.output_dir, "module_extraction_time.json")
        )
    
    @task
    def person_triplet_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['person_triplet_extraction_task'], # type: ignore[index]
            context= [self.situation_extraction_task()],  # Depends on situation task
            output_json= ExtractedPersonTriplets,
            output_file=os.path.join(self.output_dir, "module_extraction_person.json")
        )
    
    @task
    def environment_triplet_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['environment_triplet_extraction_task'], # type: ignore[index]
            context= [self.situation_extraction_task()],  # Depends on situation task
            output_json= ExtractedEnvironmentTriplets,
            output_file=os.path.join(self.output_dir, "module_extraction_environnement.json")
        )
    
    @task
    def challenge_triplet_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['challenge_triplet_extraction_task'], # type: ignore[index]
            context= [self.situation_extraction_task()],  # Depends on situation task
            output_json= ExtractedChallengeTriplets,
            output_file=os.path.join(self.output_dir, "module_extraction_challenge.json")
        )
    
    @task
    def activity_triplet_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['activity_triplet_extraction_task'], # type: ignore[index]
            context= [self.situation_extraction_task()],  # Depends on situation task
            output_json= ExtractedActivityTriplets,
            output_file=os.path.join(self.output_dir, "module_extraction_activity.json")
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ModuleExtraction crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,  # Sequential task execution
            verbose=True,
            knowledge_sources=[project_context],  # Project knowledge sources
        )


# Utility functions for external use
def create_module_crew_for_patient(patient_name: str) -> ModuleCrew:
    """
    Create a ModuleCrew instance configured for a specific patient
    
    Args:
        patient_name (str): Patient name
        
    Returns:
        ModuleCrew: Instance configured for the patient
    """
    return ModuleCrew(patient_name=patient_name)

def get_patient_module_output_directory(patient_name: str) -> str:
    """
    Return the output directory for a given patient
    
    Args:
        patient_name (str): Patient name
        
    Returns:
        str: Output directory path
    """
    return os.path.join(BASE_GRADIO_OUTPUT_DIR, patient_name)