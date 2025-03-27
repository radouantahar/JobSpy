from typing import List
import pandas as pd
import pdfplumber
from ollama import chat, ChatResponse  # Importer les fonctions nécessaires
from models import Skill, JobListing, AnalysisResult  # Importez vos modèles Pydantic
import asyncio
import aiohttp

# Définir le prompt système
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

# Fonction pour interroger le modèle Ollama
async def query_ollama(prompt: str) -> str:
    response: ChatResponse = await chat(model='llama3.2', messages=[
        {
            'role': 'system',
            'content': SYSTEM_PROMPT,
        },
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response.message.content  # Retourner le contenu du message
    

async def evaluate_job_description(description: str) -> tuple[str, str]:
    prompt = f"À partir de la description suivante, identifiez en un mot le rôle du poste proposé et le niveau hiérarchique de l'offre d'emploi :\nDescription : {description}\nRôle :\nNiveau hiérarchique :"
    
    async with aiohttp.ClientSession() as session:
        response = await query_ollama(prompt)
    
    print(f"Réponse du modèle : {response}")

    try:
        lines = response.splitlines()
        job_function = lines[0].split(":")[1].strip() if len(lines) > 0 else "Inconnu"
        job_level = lines[1].split(":")[1].strip() if len(lines) > 1 else "Inconnu"
    except IndexError:
        job_function = "Inconnu"
        job_level = "Inconnu"

    return job_function, job_level

def read_job_listings(file_path: str) -> List[JobListing]:
    job_listings_df = pd.read_csv(file_path)
    #print(job_listings_df.columns)  # Afficher les colonnes pour le débogage

    # Remplacer les NaN par des chaînes vides ou des valeurs par défaut
    job_listings_df.fillna('', inplace=True)

    return [
        JobListing(
            id=row['id'],
            site=row['site'],
            job_url=row['job_url'],
            title=row['title'],
            company=row['company'],
            job_type=row['job_type'] or "Inconnu",  # Valeur par défaut
            date_posted=row['date_posted'],
            description=row['description'],
            min_amount=float(row['min_amount']) if row['min_amount'] else None,
            max_amount=float(row['max_amount']) if row['max_amount'] else None,
            currency=row['currency'],
            is_remote=True if str(row['is_remote']).lower() == 'true' else False,  # Convertir en booléen
            job_level=row['job_level'] or "Inconnu",  # Valeur par défaut
            job_function=row.get('job_function', "Inconnu")  # Valeur par défaut
        ) for index, row in job_listings_df.iterrows()
    ]

def read_cv(file_path: str) -> str:
    cv_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            cv_text += page.extract_text() + "\n"
    return cv_text

def extract_skills(text: str) -> List[Skill]:
    prompt = f"Extraire les compétences du texte suivant : {text}"
    skills_text = query_ollama(prompt)
    skills = [Skill(name=skill.strip()) for skill in skills_text.split(",") if skill.strip()]
    return skills

def calculate_similarity(cv_text: str, job_description: str) -> float:
    prompt = f"Sur une échelle de 0 à 100, évaluez la similarité entre le CV suivant et la description de l'offre :\nCV : {cv_text}\nDescription de l'offre : {job_description}\nScore de similarité :"
    similarity_score_text = query_ollama(prompt)
    # Assurez-vous que le texte retourné est un nombre
    try:
        return float(similarity_score_text.strip())
    except ValueError:
        print(f"Erreur de conversion : {similarity_score_text}")
        return 0.0  # Ou une autre valeur par défaut

def analyze_strengths_weaknesses(job_listings: List[JobListing], cv_text: str) -> List[AnalysisResult]:
    cv_skills = extract_skills(cv_text)
    print(f"CV Skills: {[skill.name for skill in cv_skills]}")  # Ajoutez cette ligne pour le débogage
    analysis_results = []

    for job in job_listings:
        job_skills = extract_skills(job.description)
        print(f"Job Skills for {job.title}: {[skill.name for skill in job_skills]}")  # Ajoutez cette ligne pour le débogage

        strengths = [skill.name for skill in cv_skills if skill.name in [j.name for j in job_skills]]
        weaknesses = [skill.name for skill in job_skills if skill.name not in [j.name for j in cv_skills]]

        # Calculer le score de similarité
        matching_score = calculate_similarity(cv_text, job.description)

        analysis_results.append(AnalysisResult(
            job_title=job.title,
            strengths=strengths,
            weaknesses=weaknesses,
            matching_score=matching_score
        ))

    return analysis_results

def save_scored_jobs(analysis_results: List[AnalysisResult], output_file: str):
    # Convertir les résultats en DataFrame pour l'enregistrement
    df = pd.DataFrame([result.model_dump() for result in analysis_results])
    df.to_csv(output_file, index=False)

async def analyze_job_listings(job_listings: List[JobListing]) -> List[JobListing]:
    tasks = []
    for job in job_listings:
        tasks.append(evaluate_job_description(job.description))
    
    results = await asyncio.gather(*tasks)
    
    for job, (job_function, job_level) in zip(job_listings, results):
        job.job_function = job_function
        job.job_level = job_level
    
    return job_listings

def filter_job_listings(job_listings: List[JobListing], job_type: str) -> List[JobListing]:
    return [job for job in job_listings if job.job_type.lower() == job_type.lower()]

# Exemple d'utilisation
async def main():
    job_listings = read_job_listings("offres_emploi.csv")  # Remplacez par le chemin de votre fichier CSV
    job_listings = await analyze_job_listings(job_listings)  # Évaluer les descriptions
    for job in job_listings:
        print(f"Titre : {job.title}, Fonction : {job.job_function}, Niveau : {job.job_level}")

# Exécuter le programme
if __name__ == "__main__":
    asyncio.run(main())