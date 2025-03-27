import pandas as pd
import ollama
import json
import re
import csv

class JobMatcher:
    def __init__(self, cv_path, jobs_csv_path):
        """
        Initialise le matcheur d'emploi
        
        Args:
            cv_path (str): Chemin vers le fichier PDF du CV
            jobs_csv_path (str): Chemin vers le fichier CSV des offres d'emploi
        """
        self.cv_text = self._extract_cv_text(cv_path)
        
        # Charger le CSV des offres d'emploi avec une meilleure gestion des encodages
        try:
            self.jobs_df = pd.read_csv(jobs_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Essayer avec un autre encodage si utf-8 échoue
            try:
                self.jobs_df = pd.read_csv(jobs_csv_path, encoding='latin-1')
            except:
                # Dernier recours
                self.jobs_df = pd.read_csv(jobs_csv_path, encoding='cp1252')
        
        # Vérifier et assurer la présence des colonnes nécessaires
        required_columns = ['title', 'description', 'company']
        
        for col in required_columns:
            if col not in self.jobs_df.columns:
                raise ValueError(f"Le fichier CSV doit contenir une colonne '{col}'")
        
        # Ajouter des colonnes manquantes si nécessaire
        if 'location' not in self.jobs_df.columns:
            self.jobs_df['location'] = 'Non spécifié'
            
        if 'job_url' not in self.jobs_df.columns:
            self.jobs_df['job_url'] = ''
            
        # Extraire les mots-clés du CV pour une utilisation ultérieure
        self.cv_keywords = self._extract_cv_keywords()
        
    def _extract_cv_text(self, cv_path):
        """
        Extrait le texte du CV
        
        Args:
            cv_path (str): Chemin vers le fichier PDF
        
        Returns:
            str: Texte extrait du CV
        """
        import PyPDF2
        
        with open(cv_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in reader.pages])
    
    def _preprocess_text(self, text):
        """
        Prétraite le texte pour l'analyse
        
        Args:
            text (str): Texte à prétraiter
        
        Returns:
            str: Texte nettoyé
        """
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux et les chiffres
        text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', text)
        return text
    
    def _extract_cv_keywords(self):
        """
        Utilise Llama pour générer un résumé concis du CV
        
        Returns:
            str: Résumé concis du CV
        """
        # Limiter la longueur du CV pour le prompt (jusqu'à 2000 caractères pour plus de contexte)
        cv_excerpt = self.cv_text[:3000]
        
        # Créer un prompt pour générer un résumé concis
        extract_prompt = f"""
        Voici un extrait d'un CV:
        
        {cv_excerpt}
        
        Génère un résumé concis (environ 500 mots) de ce profil professionnel, en mettant en évidence:
        - Le niveau d'expérience et le domaine principal
        - Les compétences techniques principales
        - Les formations et diplômes pertinents
        - Toute information distinctive qui pourrait être importante
        
        Le résumé doit être cohérent, professionnel et facilement comparable à des offres d'emploi.
        """
        
        try:
            # Utiliser Llama pour générer le résumé
            response = ollama.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': extract_prompt}]
            )
            
            # Récupérer le résumé
            cv_summary = response['message']['content'].strip()
            
            # print(f"Résumé généré: {cv_summary[:300]}...")
            print(f"Résumé généré: {cv_summary}...")
            return cv_summary
            
        except Exception as e:
            print(f"Erreur lors de la génération du résumé: {str(e)}")
            # Fallback si la génération échoue
            return "Résumé du CV non disponible"
    
    def analyze_job_match(self, job_description):
        """
        Analyse la correspondance entre le CV et une offre d'emploi
        
        Args:
            job_description (str): Description de l'offre d'emploi
        
        Returns:
            dict: Résultats de l'analyse de correspondance
        """
        # Utiliser le résumé déjà généré et stocké dans l'instance
        cv_summary = self.cv_keywords
        
        # Limiter la taille de la description de poste (max 800 caractères)
        job_summary = job_description[:2000] if len(job_description) > 2000 else job_description
        
        # Utilisation d'Ollama pour l'analyse sémantique avec un prompt adapté au résumé
        prompt = f"""
        Analyse la correspondance entre ce résumé de CV et cette offre d'emploi:

        Résumé du CV:
        {cv_summary}

        Offre d'emploi:
        {job_summary}

        Fournis les éléments suivants:
        - Score de correspondance (0-100) basé sur l'adéquation globale du profil
        - Compétences et qualifications correspondantes (liste courte)
        - Écarts potentiels ou compétences manquantes (liste courte)
        - Recommendations (liste courte) 

        Réponds en JSON strict: {{"score": N, "compétences_clés": ["comp1", "comp2"], "écarts_potentiels": ["écart1"], "recommendations":["reco1"]}}
        """
        
        try:
            response = ollama.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Extraire uniquement la partie JSON de la réponse
            content = response['message']['content']
            json_match = re.search(r'{.*}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                match_data = json.loads(json_str)
            else:
                # Fallback si le format JSON n'est pas détecté
                match_data = {
                    "score": 0,
                    "compétences_clés": ['xyz'],
                    "écarts_potentiels": [],
                    "recommendations":[]
                }
            
            return match_data
        
        except Exception as e:
            print(f"Erreur lors de l'analyse: {str(e)}")
            return {
                "error": str(e),
                "score": 0,
                "compétences_clés": [],
                "écarts_potentiels": [],
                "recommendations":[]
            }
    
    def evaluate_all_jobs(self):
        """
        Évalue toutes les offres d'emploi et génère un CSV de résultats
        
        Returns:
            str: Chemin du fichier CSV généré
        """
        # Préparer les résultats
        results = []
        total_jobs = len(self.jobs_df)
        
        # Évaluer chaque offre d'emploi avec barre de progression
        for index, job in self.jobs_df.iterrows():
            # Afficher la progression
            print(f"Analyse de l'offre {index+1}/{total_jobs}: {job['title']}")
            
            # Combine title and description for analysis
            full_job_text = f"{job['title']} {job['description']}"
            
            # Analyse de correspondance
            match_result = self.analyze_job_match(full_job_text)
            
            # Ajouter les détails du job à l'analyse
            result_entry = {
                'job_title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'job_url': job.get('job_url', ''),
                'match_score': match_result.get('score', 0),
                'key_skills': json.dumps(match_result.get('compétences_clés', []), ensure_ascii=False),
                'potential_gaps': json.dumps(match_result.get('écarts_potentiels', []), ensure_ascii=False),
                'recommendations': json.dumps(match_result.get('recommandations', []), ensure_ascii=False)
            }
            
            # Ajouter les recommandations si présentes
            # if 'recommandations' in match_result:
            #    result_entry['recommendations'] = json.dumps(match_result.get('recommandations', []), ensure_ascii=False)
            
            results.append(result_entry)
        
        # Chemin de sortie pour le CSV
        output_path = 'job_match_evaluation.csv'
        
        # Créer un DataFrame et sauvegarder en CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # UTF-8 avec BOM pour meilleure compatibilité Excel
        
        print(f"📊 Évaluation complète des {total_jobs} offres d'emploi sauvegardée dans {output_path}")
        return output_path

def main():
    # Chemins des fichiers (à adapter)
    CV_PATH = 'RT_Resume_CV.pdf'
    JOBS_CSV_PATH = 'offres_emploi_L.csv'
    
    print("🔍 Démarrage de l'analyse de compatibilité CV-emplois")
    print(f"CV: {CV_PATH}")
    print(f"Offres d'emploi: {JOBS_CSV_PATH}")
    
    try:
        # Initialiser le matcheur
        matcher = JobMatcher(CV_PATH, JOBS_CSV_PATH)
        print(f"✅ CV chargé: {len(matcher.cv_text)} caractères")
        print(f"✅ Fichier d'offres chargé: {len(matcher.jobs_df)} offres trouvées")
        
        # Extraire les mots-clés du CV avant l'analyse
        print("\n📑 Extraction des mots-clés du CV...")
        
        # Évaluer toutes les offres et générer un CSV
        print("\n⏳ Évaluation des offres d'emploi en cours...")
        output_path = matcher.evaluate_all_jobs()
        
        print(f"\n✅ Processus terminé avec succès!")
        print(f"📊 Résultats disponibles dans: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()