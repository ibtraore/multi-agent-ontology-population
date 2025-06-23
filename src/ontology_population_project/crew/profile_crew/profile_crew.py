from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os 
from src.ontology_population_project.tools.parsing_tool import ParsingTool
from dotenv import load_dotenv
from src.ontology_population_project.pydantic_types.profile_types import ExtractedTriplets
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


load_dotenv()
os.environ["OTEL_SDK_DISABLED"] = "true"

# Base directory configuration (will be modified dynamically)
BASE_GRADIO_OUTPUT_DIR = "src/ontology_population_project/agent-output/Gradio"
os.makedirs(BASE_GRADIO_OUTPUT_DIR, exist_ok=True)

# Global variable for output directory (will be set dynamically)
current_output_dir = None

def set_output_directory(patient_name: str) -> str:
    """Configure output directory for a given patient"""
    global current_output_dir
    current_output_dir = os.path.join(BASE_GRADIO_OUTPUT_DIR, patient_name)
    os.makedirs(current_output_dir, exist_ok=True)
    return current_output_dir

def get_current_output_dir() -> str:
    """Return the currently configured output directory"""
    global current_output_dir
    if current_output_dir is None:
        # Default directory if no patient is defined
        current_output_dir = os.path.join(BASE_GRADIO_OUTPUT_DIR, "default_patient")
        os.makedirs(current_output_dir, exist_ok=True)
    return current_output_dir

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
Ce projet vise à transformer automatiquement des rapports médicaux non structurés en connaissances cliniques formalisées, 
représentées sous forme de triplets RDF compatibles avec une ontologie médicale.

Pour cela, un pipeline de quatre agents spécialisés est mis en place :

L'agent d'extraction clinique identifie les éléments importants (diagnostics, symptômes) dans un texte médical brut.

L'agent d'interprétation filtre ces éléments pour ne garder que ceux qui sont confirmés
 (pas de spéculation ou de mention négative).

L'agent d'extraction RDF convertit les éléments validés en triplets RDF respectant les règles de l'ontologie 
(relations autorisées, noms d'instances, types).

"""

project_context = StringKnowledgeSource(
    content=about_project
)

@CrewBase
class ProfileCrew():
    """Medical Ontology Parsing Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    llama_parser_tool = ParsingTool()
    
    def __init__(self, patient_name: str = None):
        """
        Initialize ProfileCrew with a patient name to define dynamic output directory
        
        Args:
            patient_name (str): Patient name to create dynamic output directory
        """
        super().__init__()
        if patient_name:
            self.output_dir = set_output_directory(patient_name)
        else:
            self.output_dir = get_current_output_dir()

    def set_patient_output_dir(self, patient_name: str):
        """
        Update output directory for a new patient
        
        Args:
            patient_name (str): Patient name
        """
        self.output_dir = set_output_directory(patient_name)

    @agent
    def parsing_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['parsing_agent'], # type: ignore[index]
            tools=[self.llama_parser_tool],
            llm= llm,
            verbose=True,
        )

    @agent
    def interpreting_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['interpreting_agent'], # type: ignore[index]
            verbose=True,
            llm= llm,
        )

    @agent
    def extracting_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['extracting_agent'], # type: ignore[index]
            llm= llm,
            verbose=True,
        )

    @task
    def parsing_task(self) -> Task:
        return Task(
            config=self.tasks_config['parsing_task'], # type: ignore[index]
            output_file=os.path.join(self.output_dir, "profile_parsing.txt")
        )

    @task
    def interpreting_task(self) -> Task:
        return Task(
            config=self.tasks_config['interpreting_task'], # type: ignore[index]
            output_file=os.path.join(self.output_dir, "profile_interpreting.txt")
        )
    
    @task
    def extracting_task(self) -> Task:
        return Task(
            config=self.tasks_config['extracting_task'], # type: ignore[index]
            output_json=ExtractedTriplets,
            output_file=os.path.join(self.output_dir, "profile_extraction.json")
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Medical Ontology Processing crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[project_context]
        )


# Utility functions for external use
def create_profile_crew_for_patient(patient_name: str) -> ProfileCrew:
    """
    Create a ProfileCrew instance configured for a specific patient
    
    Args:
        patient_name (str): Patient name
        
    Returns:
        ProfileCrew: Instance configured for the patient
    """
    return ProfileCrew(patient_name=patient_name)

def get_patient_output_directory(patient_name: str) -> str:
    """
    Return the output directory for a given patient
    
    Args:
        patient_name (str): Patient name
        
    Returns:
        str: Output directory path
    """
    return os.path.join(BASE_GRADIO_OUTPUT_DIR, patient_name)