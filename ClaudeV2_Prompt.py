import pandas as pd
import json
import re
import time
import gc
import os
import sys
import argparse
import requests
from datetime import datetime
import ollama


# Vérification des dépendances requises
try:
    import ollama
except ImportError:
    print("❌ Le package 'ollama' n'est pas installé. Veuillez l'installer avec 'pip install ollama'")
    sys.exit(1)

try:
    import PyPDF2
except ImportError:
    print("❌ Le package 'PyPDF2' n'est pas installé. Veuillez l'installer avec 'pip install PyPDF2'")
    sys.exit(1)


class JobMatcher:
    def __init__(self, cv_path, nocodb_url, nocodb_token, table_name, ollama_host=None, model_name="llama3.2"):
        """
        Initialise le matcheur d'emploi en se basant sur NocoDB.
        
        Args:
            cv_path (str): Chemin vers le fichier PDF du CV.
            nocodb_url (str): URL de l'instance NocoDB.
            nocodb_token (str): Token API pour NocoDB.
            table_name (str): Nom de la table contenant les offres d'emploi.
            ollama_host (str, optional): URL de l'hôte Ollama.
            model_name (str, optional): Nom du modèle Ollama.
        """
        # Connexion à Ollama
        self.ollama_host = ollama_host
        self.model_name = model_name

        if ollama_host:
            ollama.host = ollama_host
            print(f"✅ Connexion à Ollama configurée sur: {ollama_host}")

        print(f"🤖 Utilisation du modèle: {model_name}")

        # Connexion à NocoDB
        self.nocodb_url = nocodb_url
        self.nocodb_token = nocodb_token
        self.table_name = table_name

        # Vérifier que le CV existe
        if not os.path.exists(cv_path):
            raise FileNotFoundError(f"Le fichier CV '{cv_path}' n'existe pas")

        # Extraire le texte du CV
        self.cv_text = self._extract_cv_text(cv_path)
        print("✅ Texte du CV extrait.")

        # Extraire les compétences clés (mots-clés requis) depuis le CV
        self.extracted_keywords = self.extract_required_keywords()
        if self.extracted_keywords:
            print("✅ Mots-clés extraits depuis le CV :", self.extracted_keywords)
        else:
            print("⚠️ Aucun mot-clé n'a pu être extrait depuis le CV.")

        # Charger les offres d'emploi depuis NocoDB
        self.jobs_df = self._load_jobs_from_nocodb()

    def _extract_cv_text(self, cv_path):
        """ Extrait le texte du CV en PDF. """
        try:
            with open(cv_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return ' '.join([page.extract_text() for page in reader.pages])
        except Exception as e:
            raise Exception(f"Erreur lors de l'extraction du texte du CV: {str(e)}")

    def extract_required_keywords(self):
        """
        Utilise Ollama pour extraire les compétences clés à partir du CV.
        La réponse doit être une liste JSON.
        """
        prompt = f"""
        À partir du texte suivant, extrais une liste des compétences clés du CV.
        Réponds uniquement avec une liste JSON.
        
        CV:
        {self.cv_text[:3000]}
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            content = response['message']['content']
            keywords = json.loads(content)
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords]
            else:
                return []
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction des mots-clés du CV: {str(e)}")
            return []

    def _load_jobs_from_nocodb(self):
        """
        Charge les offres d'emploi depuis la table NocoDB.
        
        Returns:
            pd.DataFrame: DataFrame contenant les offres.
        """
        url = f"{self.nocodb_url}/api/v2/tables/{self.table_name}/records"
        headers = {"xc-token": self.nocodb_token}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            records = response.json().get('list', [])
            if not records:
                raise ValueError("❌ Aucune offre trouvée dans NocoDB !")
            jobs_df = pd.DataFrame(records)
            print(f"✅ {len(jobs_df)} offres d'emploi chargées depuis NocoDB")
            return jobs_df
        except requests.exceptions.RequestException as e:
            raise Exception(f"❌ Erreur lors du chargement des offres depuis NocoDB : {str(e)}")

    def _save_match_results_to_nocodb(self, job_id, match_result):
        """
        Sauvegarde les résultats d'analyse dans NocoDB.
        Args:
            job_id (int): ID de l'offre d'emploi dans NocoDB.
            match_result (dict): Résultats de l'analyse de correspondance.
        """
        url = f"{self.nocodb_url}/api/v2/tables/{self.table_name}/records/{job_id}"
        headers = {
            "xc-token": self.nocodb_token,
            "Content-Type": "application/json"
        }
        update_data = {
            "match_score": match_result.get('score', 0),
            "key_skills": json.dumps(match_result.get('compétences_clés', []), ensure_ascii=False),
            "potential_gaps": json.dumps(match_result.get('écarts_potentiels', [])),
            "recommendations": json.dumps(match_result.get('recommendations', []))
        }
        try:
            response = requests.patch(url, headers=headers, json=update_data)
            response.raise_for_status()
            print(f"✅ Résultats enregistrés dans NocoDB pour job ID {job_id}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur lors de l'enregistrement des résultats dans NocoDB: {str(e)}")

    def filter_jobs(self, locations=None, contract_types=None, min_salary=None, required_keywords=None, 
                    exclude_keywords=None, required_education=None):
        """
        Filtre les offres d'emploi selon plusieurs critères.
        
        Args:
            locations (list, optional): Localisations acceptées.
            contract_types (list, optional): Types de contrats acceptés.
            min_salary (int, optional): Salaire minimum acceptable.
            required_keywords (list, optional): Mots-clés requis.
            exclude_keywords (list, optional): Mots-clés à exclure.
            required_education (list, optional): Niveaux d'études requis (ex: ["Bac+5", "ingénieur", "master"]).
        
        Returns:
            pd.DataFrame: DataFrame filtré.
        """
        filtered = self.jobs_df.copy()
        
        # Filtrage par localisation
        if locations:
            pattern = '|'.join([re.escape(loc.lower()) for loc in locations])
            if 'location' in filtered.columns:
                filtered = filtered[filtered['location'].str.lower().str.contains(pattern, na=False)]
        
        # Filtrage par type de contrat
        if contract_types:
            pattern = '|'.join([re.escape(ct.lower()) for ct in contract_types])
            if 'contract_type' in filtered.columns:
                filtered = filtered[filtered['contract_type'].str.lower().str.contains(pattern, na=False)]
            else:
                filtered = filtered[filtered['description'].str.lower().str.contains(pattern, na=False)]
        
        # Filtrage par salaire minimum
        if min_salary is not None:
            def salary_filter(row):
                min_amt = row.get('min_amount')
                max_amt = row.get('max_amount')
                # Si au moins une info salariale est présente et >= min_salary, conserver
                if pd.notna(min_amt):
                    try:
                        if float(min_amt) >= min_salary:
                            return True
                    except Exception:
                        pass
                if pd.notna(max_amt):
                    try:
                        if float(max_amt) >= min_salary:
                            return True
                    except Exception:
                        pass
                # Si aucune info salariale n'est renseignée, garder l'offre
                if pd.isna(min_amt) and pd.isna(max_amt):
                    return True
                return False
            filtered = filtered[filtered.apply(salary_filter, axis=1)]
        
        # Filtrage par mots-clés requis (si non fourni, on peut utiliser ceux extraits du CV)
        if not required_keywords and self.extracted_keywords:
            required_keywords = self.extracted_keywords

        if required_keywords:
            def contains_required_keywords(row):
                text = (str(row.get('title', '')) + ' ' + str(row.get('description', ''))).lower()
                for kw in required_keywords:
                    if kw.lower() not in text:
                        return False
                return True
            filtered = filtered[filtered.apply(contains_required_keywords, axis=1)]
        
        # Exclusion par mots-clés
        if exclude_keywords:
            def contains_excluded_keywords(row):
                text = (str(row.get('title', '')) + ' ' + str(row.get('description', ''))).lower()
                for kw in exclude_keywords:
                    if kw.lower() in text:
                        return True
                return False
            filtered = filtered[~filtered.apply(contains_excluded_keywords, axis=1)]
        
        # Filtrage par niveau d'études requis (ex: Bac+5, ingénieur, master)
        if required_education:
            def contains_required_education(row):
                text = (str(row.get('title', '')) + ' ' + str(row.get('description', ''))).lower()
                for edu in required_education:
                    if edu.lower() in text:
                        return True
                return False
            filtered = filtered[filtered.apply(contains_required_education, axis=1)]
        
        print(f"🔍 Filtrage terminé: {len(filtered)} offres retenues sur {len(self.jobs_df)}")
        return filtered

    def prioritize_jobs(self, df, required_keywords=None):
        """
        Calcule un score heuristique pour chaque offre et trie le DataFrame par score décroissant.
        
        Args:
            df (pd.DataFrame): DataFrame d'offres filtrées.
            required_keywords (list, optional): Mots-clés requis pour le calcul du score.
        
        Returns:
            pd.DataFrame: DataFrame trié par score.
        """
        df = df.copy()
        def score_job(row):
            text = (str(row.get('title', '')) + ' ' + str(row.get('description', ''))).lower()
            if required_keywords and len(required_keywords) > 0:
                count = sum(1 for kw in required_keywords if kw.lower() in text)
                score = (count / len(required_keywords)) * 100
            else:
                score = 50
            return score
        df['heuristic_score'] = df.apply(score_job, axis=1)
        df = df.sort_values(by='heuristic_score', ascending=False)
        print("🔍 Priorisation terminée.")
        return df

    def evaluate_all_jobs(self, locations=None, contract_types=None, min_salary=None, required_keywords=None, 
                          exclude_keywords=None, required_education=None):
        """
        Filtre, priorise et évalue les offres d'emploi, puis enregistre les résultats dans NocoDB.
        """
        # Filtrer selon les critères
        filtered_jobs = self.filter_jobs(locations, contract_types, min_salary, required_keywords, 
                                         exclude_keywords, required_education)
        if filtered_jobs.empty:
            print("❌ Aucune offre ne correspond aux critères de filtrage.")
            return
        
        # Prioriser les offres
        prioritized_jobs = self.prioritize_jobs(filtered_jobs, required_keywords)
        total_jobs = len(prioritized_jobs)
        print(f"\n⏳ Début de l'analyse de {total_jobs} offres d'emploi priorisées...")
        
        for index, job in prioritized_jobs.iterrows():
            print(f"📊 Analyse de l'offre {index+1}/{total_jobs}: {job.get('title', 'N/A')} - {job.get('company', 'N/A')}")
            match_result = self.analyze_job_match(job.get('description', ''), job.get('title', ''))
            self._save_match_results_to_nocodb(job.get('id'), match_result)
        
        print("\n✅ Analyse terminée et résultats enregistrés dans NocoDB.")

    def analyze_job_match(self, job_description, job_title):
        """
        Analyse la correspondance entre le CV et une offre d'emploi.
        
        Args:
            job_description (str): Description de l'offre d'emploi.
            job_title (str): Titre du poste.
        
        Returns:
            dict: Résultats de l'analyse de correspondance.
        """
        prompt = f"""
        # Analyse de compatibilité CV-Emploi

        ## Profil du candidat
        ```
        {self.cv_text[:3000]}
        ```

        ## Offre d'emploi
        Titre: {job_title}
        Description:
        ```
        {job_description[:2000]}
        ```

        ## Objectif
        Analyse la correspondance et attribue un score.

        ## Format JSON:
        {{
          "score": X,
          "compétences_clés": ["compétence 1", "compétence 2"],
          "écarts_potentiels": ["écart 1", "écart 2"],
          "recommendations": ["recommandation 1", "recommandation 2"]
        }}
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            content = response['message']['content']
            return json.loads(content)
        except Exception as e:
            print(f"❌ Erreur d'analyse: {str(e)}")
            return {"score": 0, "compétences_clés": [], "écarts_potentiels": [], "recommendations": []}


def main():
    parser = argparse.ArgumentParser(description="Analyse de compatibilité CV-emplois avec NocoDB")
    parser.add_argument('--cv', required=True, help='Chemin du CV en PDF')
    parser.add_argument('--nocodb-url', required=True, help='URL de NocoDB')
    parser.add_argument('--nocodb-token', required=True, help='Token API de NocoDB')
    parser.add_argument('--table-name', required=True, help='Nom de la table NocoDB')
    
    # Arguments pour le filtrage et la priorisation
    parser.add_argument('--locations', nargs='+', help='Localisations acceptées (ex: "Paris" "Lyon")')
    parser.add_argument('--contract-types', nargs='+', help='Types de contrats acceptés (ex: "CDI")')
    parser.add_argument('--min-salary', type=int, help='Salaire minimum acceptable')
    parser.add_argument('--required-keywords', nargs='+', help='Mots-clés requis (ex: "Python" "Machine Learning")')
    parser.add_argument('--exclude-keywords', nargs='+', help='Mots-clés à exclure (ex: "stage" "alternance")')
    parser.add_argument('--required-education', nargs='+', help='Niveaux d\'étude requis (ex: "Bac+5", "ingénieur", "master")')
    
    args = parser.parse_args()
    
    matcher = JobMatcher(args.cv, args.nocodb_url, args.nocodb_token, args.table_name, ollama_host=None, model_name="llama3.2")
    matcher.evaluate_all_jobs(
        locations=args.locations,
        contract_types=args.contract_types,
        min_salary=args.min_salary,
        required_keywords=args.required_keywords,
        exclude_keywords=args.exclude_keywords,
        required_education=args.required_education
    )


if __name__ == '__main__':
    main()



# python ClaudeV2_Prompt.py --cv "RT_Resume_CV.pdf" --nocodb-url "http://localhost:8080" --nocodb-token "MXnPaeY8ZY8pxPvuW6krLCgSGhQ_qRRlxH1vMbAc" --table-name "m15obttbc9p9hmf" --locations "Paris" "Reims" --contract-types "CDI" --min-salary 45000 --exclude-keywords "stage" "alternance" --required-education "Bac+5" "ingénieur" "master"

