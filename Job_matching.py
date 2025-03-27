import asyncio
import pandas as pd
import pdfplumber
import logging
import json
from typing import List
from models import Skill, JobListing, AnalysisResult  # Importer les modèles Pydantic
from ollama import chat, ChatResponse  # Importer la bibliothèque Ollama

# Configurer le logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 🔹 Prompt système pour guider Ollama
SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans l'analyse d'offres d'emploi et de CV.  
Ton objectif est d'extraire des informations utiles sur les offres et d'évaluer leur correspondance avec un CV.  

### Règles :  
1. **Reste factuel** : Ne fais aucune supposition si une information est absente.  
2. **Format des réponses** :  
   - Lorsque tu identifies le rôle et le niveau hiérarchique d'un poste, réponds en JSON :  
     ```json
     {"job_function": "Nom du rôle", "job_level": "Niveau hiérarchique"}
     ```
   - Lorsque tu évalues la similarité entre un CV et une offre, retourne uniquement un nombre entre 0 et 100.  
   - Pour l'extraction des compétences, retourne une liste de compétences sous forme de texte.  

### Exemples de sortie attendue :  
- **Rôle et niveau hiérarchique** :  
  ```json
  {"job_function": "Ingénieur logiciel", "job_level": "Senior"}
  """

async def query_ollama_async(prompt: str) -> str:
    logging.debug(f"Envoi de la requête à Ollama avec le prompt : {prompt}")
    
    try:
        response: ChatResponse = await chat(model='llama3.2', messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        
        full_response = response.message.content
        logging.debug(f"Réponse complète reçue d'Ollama : {full_response}")
        return full_response
    except Exception as e:
        logging.error(f"Erreur lors de la requête à Ollama : {e}")
        return ""

def read_job_listings(file_path: str) -> List[JobListing]:
    logging.debug(f"Lecture des offres d'emploi depuis le fichier : {file_path}")
    job_listings_df = pd.read_csv(file_path)
    job_listings_df.fillna('', inplace=True)  # Remplacer les NaN par des valeurs par défaut

    return [
        JobListing(
            id=row['id'],
            site=row['site'],
            job_url=row['job_url'],
            title=row['title'],
            company=row['company'],
            job_type=row['job_type'] or "Inconnu",
            date_posted=row['date_posted'],
            description=row['description'],
            min_amount=float(row['min_amount']) if row['min_amount'] else None,
            max_amount=float(row['max_amount']) if row['max_amount'] else None,
            currency=row['currency'],
            is_remote=str(row['is_remote']).lower() == 'true',
            job_level=row['job_level'] or "Inconnu",
            job_function=row.get('job_function', "Inconnu")
        ) for index, row in job_listings_df.iterrows()
    ]

def read_cv(file_path: str) -> str:
    logging.debug(f"Lecture du CV depuis le fichier : {file_path}")
    cv_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            cv_text += page.extract_text() + "\n"
    return cv_text

async def extract_skills(text: str) -> List[Skill]:
    prompt = f"Extraire les compétences du texte suivant : {text}"
    logging.debug(f"Extraction des compétences avec le prompt : {prompt}")
    skills_text = await query_ollama_async(prompt)
    return [Skill(name=skill.strip()) for skill in skills_text.split(",") if skill.strip()]

async def calculate_similarity(cv_text: str, job_description: str) -> float:
    prompt = f"Sur une échelle de 0 à 100, évaluez la similarité entre le CV et la description de l'offre :\nCV : {cv_text}\nDescription : {job_description}\nScore :"
    logging.debug(f"Calcul de la similarité avec le prompt : {prompt}")
    similarity_score_text = await query_ollama_async(prompt)

    try:
        score = float(similarity_score_text.strip())
        logging.debug(f"Score de similarité calculé : {score}")
        return score
    except ValueError:
        logging.error(f"Erreur de conversion : {similarity_score_text}")
        return 0.0  # Valeur par défaut

async def analyze_job_listings_async(job_listings: List[JobListing]):
    results = []

    for job in job_listings:
        prompt = f"Analyse l'offre suivante et retourne un JSON structuré :\n{job.description}"
        logging.debug(f"Analyse de l'offre : {prompt}")
        response = await query_ollama_async(prompt)
        
        # Afficher le contenu brut reçu avant le parsing
        logging.debug(f"Contenu brut reçu : {response}")
        
        data = parse_response(response)  # Utiliser la fonction de parsing
        if data:
            job.job_function = data.get("job_function", "Inconnu")
            job.job_level = data.get("job_level", "Inconnu")
            results.append(job)
        else:
            logging.error("Aucune donnée valide reçue pour l'offre.")

    return results

def parse_response(response_text: str):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logging.error(f"Erreur de décodage JSON : {response_text}")
        return None

def save_scored_jobs(analysis_results: List[AnalysisResult], output_file: str):
    df = pd.DataFrame([result.model_dump() for result in analysis_results])
    df.to_csv(output_file, index=False)
    logging.debug(f"Résultats sauvegardés dans le fichier : {output_file}")

if __name__ == "__main__":
    logging.info("Démarrage de l'analyse des offres d'emploi.")
    job_listings = read_job_listings("offres_emploi_L.csv")  # Charger les offres
    job_listings = asyncio.run(analyze_job_listings_async(job_listings))  # Analyser les offres

    # Charger et analyser le CV (décommente si besoin)
    # cv_text = read_cv("RT_Resume_CV.pdf")
    # analysis_results = asyncio.run(analyze_strengths_weaknesses(job_listings, cv_text))
    # save_scored_jobs(analysis_results, "analyse_forces_faiblesses.csv")

    for job in job_listings:
        logging.info(f"Titre : {job.title}, Fonction : {job.job_function}, Niveau : {job.job_level}")



