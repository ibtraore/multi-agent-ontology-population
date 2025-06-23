from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os 
from llama_parse import LlamaParse

output_dir = "src/ontology_population_project/agent-output/Gradio"
os.makedirs(output_dir, exist_ok=True)



#Extracting data from charts and graphs

document_structure_prompt = """
Le document à analyser est un rapport médical complexe au format PDF, généralement structuré de la manière suivante :

1. **Tableaux diagnostiques structurés**  
   - Chaque tableau correspond à un **critère diagnostique psychiatrique** écrit en titre.
   - Colonnes typiques :  
     - QP (Question Parent) et QE (Question Enseignant) — identifiants des questions (peu utiles pour la suite).  
     - Texte de la **question posée** décrivant un **signe clinique** observé.  
     - Scores donnés par 4 évaluateurs : P1 (Parent 1), P2 (Parent 2), E1 (Enseignant 1), E2 (Enseignant 2).  
   - Chaque ligne représente un **signe clinique** évalué par les 4 acteurs.  
   - Les **deux dernières lignes** du tableau présentent une **synthèse** :  
     - Nombre de critères majeurs validés  
     - T-score

   👉 Merci d’extraire fidèlement les lignes et colonnes utiles (signe clinique + scores par évaluateur + synthèse). Les identifiants QP/QE peuvent être ignorés.

2. **Figure finale : graphique en barres**  
   - Le graphique contient 11 symptômes sur l’axe horizontal.  
   - Pour chaque symptôme, on trouve 4 barres (scores) correspondant aux évaluations de :  
     - Maman  
     - Papa  
     - Enseignant 1  
     - Enseignant 2  
   - Total attendu : 44 barres (11 symptômes × 4 évaluateurs)


Merci de restituer les informations de manière lisible, hiérarchique et exploitable.
"""



parser = LlamaParse(
   result_type="markdown",  # "markdown" and "text" are available,
   
    extract_charts=True,

    auto_mode=True,

    auto_mode_trigger_on_image_in_page=True,

    auto_mode_trigger_on_table_in_page=True,
    language="fr", # Optionally you can define a language, default=en   
    verbose=True,
   user_prompt= document_structure_prompt
   
   )

class ParsingTool(BaseTool):
    name: str = "Outil de parsing PDF médical basé sur LlamaParse"
    description: str = (
        """ Cet outil extrait le contenu structuré (textes, tableaux, graphiques) à partir d’un fichier PDF médical.
            Il est spécialement conçu pour reconnaître les tableaux cliniques.

            Args:
                file_path (str): Chemin du fichier PDF à analyser.
                patient_name (str): le dossier dans le quel il faut enregistrer l'output du parseur

            Returns:
                str: Texte extrait et structuré, prêt à être nettoyé et transformé en représentation sémantique."""
    )

    def _run(self, file_path: str, patient_name:str) -> str:
        # Implementation goes here
        result_charts = parser.load_data(file_path)
        result=result_charts[0].text
        patient_dir = os.path.join(output_dir, patient_name)
        os.makedirs(patient_dir, exist_ok=True) 
        output_path = os.path.join( patient_dir, "llama_parser_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
            
        return result
    
