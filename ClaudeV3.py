import pandas as pd
import ollama
import json
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class JobMatcher:
    def __init__(self, cv_path, jobs_csv_path, max_jobs_to_analyze=50):
        """
        Initialise le matcheur d'emploi
        
        Args:
            cv_path (str): Chemin vers le fichier PDF du CV
            jobs_csv_path (str): Chemin vers le fichier CSV des offres d'emploi
            max_jobs_to_analyze (int): Nombre maximum d'offres à analyser en détail
        """
        self.cv_text = self._extract_cv_text(cv_path)
        self.max_jobs_to_analyze = max_jobs_to_analyze
        
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
        
        # Nettoyer les textes des offres d'emploi
        self.jobs_df['clean_text'] = self.jobs_df['title'] + ' ' + self.jobs_df['description']
        self.jobs_df['clean_text'] = self.jobs_df['clean_text'].apply(self._preprocess_text)
        
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
        text = str(text).lower()
        # Supprimer les caractères spéciaux et les chiffres
        text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_cv_keywords(self):
        """
        Utilise Llama pour extraire les informations clés du CV
        
        Returns:
            str: Résumé structuré du CV avec compétences clés
        """
        # Limiter la longueur du CV pour le prompt (réduire pour éviter la troncature)
        cv_excerpt = self.cv_text[:2000]
        
        # Créer un prompt plus court et concis
        extract_prompt = f"""
        Extrait de CV: {cv_excerpt}
        
        Fournis un résumé concis de ce CV avec:
        1. PROFIL: profil professionnel en 5 phrases
        2. COMPÉTENCES TECHNIQUES: liste courte
        3. EXPÉRIENCE: années et postes clés
        4. FORMATION: diplômes principaux
        5. MOTS-CLÉS: 30 mots-clés essentiels pour un matching d'emploi
        """
        
        try:
            # Utiliser Llama pour générer l'analyse
            response = ollama.chat(
                model='deepseek-r1',
                messages=[
                    {'role': 'system', 'content': 'Tu es un expert en recrutement qui extrait les informations clés des CV de façon concise.'},
                    {'role': 'user', 'content': extract_prompt}
                ]
            )
            
            # Récupérer le résumé
            cv_summary = response['message']['content'].strip()
            
            print(f"Analyse du CV générée:")
            print("----------------------------")
            print(cv_summary[:500] + "..." if len(cv_summary) > 500 else cv_summary)
            print("----------------------------")
            
            return cv_summary
            
        except Exception as e:
            print(f"Erreur lors de la génération de l'analyse du CV: {str(e)}")
            # Fallback si la génération échoue
            return "Analyse du CV non disponible"
    
    def prefilter_jobs(self, preferences=None):
        """
        Préfiltre les offres d'emploi selon des mots-clés et des préférences
        
        Args:
            preferences (dict): Dictionnaire avec les préférences
                - keywords (list): Mots-clés à rechercher
                - exclude_keywords (list): Mots-clés à exclure
                - locations (list): Lieux de travail préférés
                - min_salary (int): Salaire minimum
                - max_commute (str): Temps de trajet maximum
                - remote (bool): Travail à distance
        
        Returns:
            pd.DataFrame: DataFrame filtré des offres d'emploi
        """
        # Valeurs par défaut si aucune préférence n'est fournie
        if preferences is None:
            preferences = {}
        
        # Extraire les mots-clés du CV pour le filtrage
        cv_cleaned = self._preprocess_text(self.cv_text)
        
        # Utiliser TF-IDF pour comparer le CV et les offres d'emploi
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        try:
            # Créer un corpus avec le CV et les descriptions de poste
            corpus = [cv_cleaned] + list(self.jobs_df['clean_text'])
            
            # Transformer le corpus en matrice TF-IDF
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Calculer la similarité cosinus entre le CV et chaque offre d'emploi
            cv_vector = tfidf_matrix[0:1]
            job_vectors = tfidf_matrix[1:]
            cosine_similarities = cosine_similarity(cv_vector, job_vectors).flatten()
            
            # Ajouter les scores de similarité au DataFrame
            self.jobs_df['similarity_score'] = cosine_similarities
            
        except Exception as e:
            print(f"Erreur lors du calcul de similarité TF-IDF: {str(e)}")
            # Fallback: créer une colonne de similarité avec des valeurs aléatoires
            self.jobs_df['similarity_score'] = np.random.uniform(0, 1, size=len(self.jobs_df))
        
        # Appliquer les filtres de préférences spécifiques
        filtered_df = self.jobs_df.copy()
        
        # Filtrer par mots-clés (si spécifiés)
        if 'keywords' in preferences and preferences['keywords']:
            pattern = '|'.join(preferences['keywords'])
            mask = (filtered_df['title'].str.contains(pattern, case=False, na=False) | 
                   filtered_df['description'].str.contains(pattern, case=False, na=False))
            filtered_df = filtered_df[mask]
        
        # Exclure les offres contenant certains mots-clés (si spécifiés)
        if 'exclude_keywords' in preferences and preferences['exclude_keywords']:
            pattern = '|'.join(preferences['exclude_keywords'])
            mask = ~(filtered_df['title'].str.contains(pattern, case=False, na=False) | 
                    filtered_df['description'].str.contains(pattern, case=False, na=False))
            filtered_df = filtered_df[mask]
        
        # Filtrer par lieu (si spécifié)
        if 'locations' in preferences and preferences['locations']:
            pattern = '|'.join(preferences['locations'])
            mask = filtered_df['location'].str.contains(pattern, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Filtrer par travail à distance (si spécifié)
        if 'remote' in preferences and preferences['remote']:
            mask = (filtered_df['description'].str.contains('remote|télétravail|à distance', 
                                                           case=False, na=False) |
                   filtered_df['title'].str.contains('remote|télétravail|à distance', 
                                                    case=False, na=False))
            filtered_df = filtered_df[mask]
        
        # Trier par score de similarité
        filtered_df = filtered_df.sort_values(by='similarity_score', ascending=False)
        
        # Limiter le nombre d'offres
        limited_df = filtered_df.head(self.max_jobs_to_analyze)
        
        print(f"Préfiltrage terminé: {len(filtered_df)} offres sur {len(self.jobs_df)} correspondent aux critères")
        print(f"Analyse détaillée limitée aux {len(limited_df)} meilleures offres")
        
        return limited_df
    
    def analyze_job_match(self, job_description, job_title):
        """
        Analyse la correspondance entre le CV et une offre d'emploi
        
        Args:
            job_description (str): Description de l'offre d'emploi
            job_title (str): Titre du poste
        
        Returns:
            dict: Résultats de l'analyse de correspondance
        """
        # Utiliser le résumé déjà généré et stocké dans l'instance
        cv_summary = self.cv_keywords
        
        # Limiter davantage la taille de la description de poste pour éviter la troncature
        job_summary = job_description[:1000] if len(job_description) > 1000 else job_description
        
        # Prompt simplifié et plus court
        prompt = f"""
        CV: {cv_summary[:500]}
        
        Offre: {job_title}
        Description: {job_summary[:1000]}
        
        Évalue la compatibilité CV-emploi. Réponds au format JSON uniquement:
        {{
          "score": X (0-100),
          "competences_cles": ["compétence 1", "compétence 2", "compétence 3"],
          "ecarts_potentiels": ["écart 1", "écart 2"],
          "recommendations": ["recommandation 1", "recommandation 2"]
        }}
        """
        
        try:
            # Utiliser Llama avec un message système court
            response = ollama.chat(
                model='deepseek-r1',
                messages=[
                    {'role': 'system', 'content': 'Expert en recrutement. Réponds en JSON uniquement.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            # Extraire uniquement la partie JSON de la réponse
            content = response['message']['content']
            
            # Méthode améliorée pour extraire le JSON
            content = content.strip()
            
            # Si la réponse commence par ```json et se termine par ```, extraire le contenu
            if content.startswith('```json') and content.endswith('```'):
                content = content[7:-3].strip()
            # Si la réponse commence par ``` et se termine par ```, extraire le contenu
            elif content.startswith('```') and content.endswith('```'):
                content = content[3:-3].strip()
                
            try:
                match_data = json.loads(content)
            except json.JSONDecodeError:
                # Tentative supplémentaire avec regex pour extraire le JSON
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    try:
                        match_data = json.loads(json_match.group(1))
                    except:
                        raise ValueError("Format JSON non valide dans la réponse")
                else:
                    raise ValueError("Aucun format JSON détecté dans la réponse")
            
            # Normaliser les clés pour être cohérent
            normalized_data = {
                "score": match_data.get("score", 0),
                "compétences_clés": match_data.get("competences_cles", []),
                "écarts_potentiels": match_data.get("ecarts_potentiels", []),
                "recommendations": match_data.get("recommendations", [])
            }
            
            return normalized_data
        
        except Exception as e:
            print(f"Erreur lors de l'analyse: {str(e)}")
            return {
                "error": str(e),
                "score": 0,
                "compétences_clés": [],
                "écarts_potentiels": [],
                "recommendations": []
            }
    
    def evaluate_jobs(self, filtered_jobs=None):
        """
        Évalue les offres d'emploi préfiltrées et génère un CSV de résultats
        
        Args:
            filtered_jobs (pd.DataFrame): DataFrame des offres d'emploi préfiltrées
        
        Returns:
            str: Chemin du fichier CSV généré
        """
        # Utiliser les offres filtrées si fournies, sinon utiliser toutes les offres
        if filtered_jobs is None:
            jobs_to_evaluate = self.jobs_df
        else:
            jobs_to_evaluate = filtered_jobs
        
        # Préparer les résultats
        results = []
        total_jobs = len(jobs_to_evaluate)
        
        # Évaluer chaque offre d'emploi avec barre de progression
        for index, job in jobs_to_evaluate.iterrows():
            # Afficher la progression
            print(f"Analyse de l'offre {len(results)+1}/{total_jobs}: {job['title']}")
            
            # Analyse de correspondance avec passage du titre séparé
            match_result = self.analyze_job_match(job['description'], job['title'])
            
            # Créer une copie de la ligne entière du job dans un dictionnaire
            result_entry = job.to_dict()
            
            # Ajouter les résultats de l'analyse à ce dictionnaire
            result_entry.update({
                'match_score': match_result.get('score', 0),
                'key_skills': json.dumps(match_result.get('compétences_clés', []), ensure_ascii=False),
                'potential_gaps': json.dumps(match_result.get('écarts_potentiels', []), ensure_ascii=False),
                'recommendations': json.dumps(match_result.get('recommendations', []), ensure_ascii=False)
            })
            
            results.append(result_entry)
        
        # Trier les résultats par score de correspondance (du plus élevé au plus bas)
        sorted_results = sorted(results, key=lambda x: x['match_score'], reverse=True)
        
        # Chemin de sortie pour le CSV
        output_path = 'job_match_evaluation.csv'
        
        # Créer un DataFrame et sauvegarder en CSV
        results_df = pd.DataFrame(sorted_results)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"📊 Évaluation complète des {total_jobs} offres d'emploi sauvegardée dans {output_path}")
        
        # Afficher un résumé des meilleurs matchs
        top_matches = sorted_results[:5] if len(sorted_results) >= 5 else sorted_results
        print("\n🏆 Top des correspondances:")
        for i, match in enumerate(top_matches):
            print(f"{i+1}. {match['title']} - {match['company']} (Score: {match['match_score']})")
        
        return output_path


def main():
    # Chemins des fichiers (à adapter)
    CV_PATH = 'RT_Resume_CV.pdf'
    JOBS_CSV_PATH = 'CSV/all_jobs_combined.csv'
    
    # Nombre maximum d'offres à analyser en détail (ajuster selon vos besoins)
    MAX_JOBS = 5
    
    # Préférences de filtrage (à personnaliser)
    preferences = {
    "keywords": [
        "ingénieur méthodes", "chef de projet", "PMO", "pilote industrialisation", 
        "amélioration continue", "lean manufacturing", "gestion de projet", 
        "industrialisation", "PLM", "EBOM", "BPMN", "6 sigma", "green belt",
        "analyse de valeur", "processus", "flux", "implantation", "robotique",
        "JIRA", "Agile", "SAFe", "PowerBI", "Enovia", 
        "management d'équipe", "multiculturel", "pluridisciplinaire"
    ],
    "exclude_keywords": [
        "vice président", "VP", "directeur général", "DG", "CTO", "CDO", 
        "junior", "stagiaire", "alternance", "stage"
    ],
    "locations": [
        "Paris", "Reims", "Marne", "remote", "télétravail", "à distance", "hybride"
    ],
    "remote": 'true',
    "industries": [
        "luxe", "tech", "industriel", "robotique", "automobile", "aéronautique",
        "manufacture", "production", "high-tech", "horlogerie", "joaillerie"
    ],
    "company_sizes": [
        "PME", "ETI", "grand groupe"
    ]
}
    
    print("🔍 Démarrage de l'analyse de compatibilité CV-emplois avec préfiltrage")
    print(f"CV: {CV_PATH}")
    print(f"Offres d'emploi: {JOBS_CSV_PATH}")
    print(f"Préférences: {json.dumps(preferences, ensure_ascii=False)}")
    
    try:
        # Initialiser le matcheur
        matcher = JobMatcher(CV_PATH, JOBS_CSV_PATH, max_jobs_to_analyze=MAX_JOBS)
        print(f"✅ CV chargé: {len(matcher.cv_text)} caractères")
        print(f"✅ Fichier d'offres chargé: {len(matcher.jobs_df)} offres trouvées")
        
        # Préfiltrer les offres selon les préférences
        print("\n📊 Préfiltrage des offres d'emploi selon vos préférences...")
        filtered_jobs = matcher.prefilter_jobs(preferences)
        
        # Évaluer les offres préfiltrées et générer un CSV
        print("\n⏳ Évaluation détaillée des offres préfiltrées en cours...")
        output_path = matcher.evaluate_jobs(filtered_jobs)
        
        print(f"\n✅ Processus terminé avec succès!")
        print(f"📊 Résultats disponibles dans: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()