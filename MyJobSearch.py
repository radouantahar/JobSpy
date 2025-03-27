import os
import fitz
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from jobspy import scrape_jobs
import re
import time
import math
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from pandera import Check, Column, DataFrameSchema
import threading


# ======================
# CONFIGURATION
# ======================
CONFIG = {
    "resume_path": "RT_Resume_CV.pdf",
    "nocodb": {
        "api_url": "http://localhost:8080",
        "api_token": "MXnPaeY8ZY8pxPvuW6krLCgSGhQ_qRRlxH1vMbAc",
        "table_name": "mt8wyajtfrlxix5",
        "expected_columns": [
            "id1", "site", "job_url", "job_url_direct", "title", "company",
            "location", "date_posted", "job_type", "salary_source", "interval",
            "min_amount", "max_amount", "currency", "is_remote", "job_level",
            "job_function", "listing_type", "emails", "description",
            "company_industry", "company_url", "company_logo",
            "company_url_direct", "company_addresses", "company_num_employees",
            "company_revenue", "company_description"
        ]
    },
    "search_params": {
        "locations": [],     
        "search_terms": [
            "Chef de Projet Industrialisation",
            "Responsable Méthodes Industrielles",
            "Chef de Projet"
        ],
        "max_results": 50,
        "max_days_old": 7,
        "retry_attempts": 3,
        "chunk_size": 100
    },
    "ai": {
        "spacy_model": "fr_core_news_sm",
        "sentence_model": "distiluse-base-multilingual-cased-v1",
        "local_llm": {
            "base_url": "http://localhost:11434",
            "model": "llama3.2",
            "temperature": 0.3,
            "max_tokens": 2000
        },
        "match_threshold": 0.60
    }
}



# ======================
# INITIALIZATION
# ======================
nlp = spacy.load(CONFIG["ai"]["spacy_model"])
sbert_model = SentenceTransformer(CONFIG["ai"]["sentence_model"])

# tps_transport_L_Test

with open('tps_transport_L.json', 'r', encoding='utf-8') as f:
    TRANSPORT_DATA = json.load(f)

# ======================
# CORE COMPONENTS
# ======================

def process_relative_dates(df):
    """Process relative date formats like '3 days ago' into actual datetime objects."""
    if 'date_posted' not in df.columns:
        return df
        
    # Get today's date for relative calculations (date only, no time component)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    def parse_relative_date(date_str):
        if not isinstance(date_str, str):
            return date_str
            
        date_str = date_str.lower().strip()
        
        # Handle common patterns
        if date_str == "today" or date_str == "just posted" or date_str == "just now":
            return today
        elif date_str == "yesterday":
            return today - timedelta(days=1)
        elif "hour" in date_str or "hr" in date_str:
            # Extract number of hours (e.g., "2 hours ago")
            hours = int(re.search(r"(\d+)", date_str).group(1)) if re.search(r"(\d+)", date_str) else 1
            return today - timedelta(hours=hours)
        elif "day" in date_str:
            # Extract number of days (e.g., "3 days ago")
            days = int(re.search(r"(\d+)", date_str).group(1)) if re.search(r"(\d+)", date_str) else 1
            return today - timedelta(days=days)
        elif "week" in date_str:
            # Extract number of weeks (e.g., "2 weeks ago")
            weeks = int(re.search(r"(\d+)", date_str).group(1)) if re.search(r"(\d+)", date_str) else 1
            return today - timedelta(weeks=weeks)
        elif "month" in date_str:
            # Extract number of months (approximately)
            months = int(re.search(r"(\d+)", date_str).group(1)) if re.search(r"(\d+)", date_str) else 1
            return today - timedelta(days=30*months)
        
        # Let pd.to_datetime handle parseable formats
        return date_str
    
    # Apply the function to the date_posted column
    df['date_posted'] = df['date_posted'].apply(parse_relative_date)
    return df

class TransportAnalyzer:
    @staticmethod
    def parse_duration(duration_str: str) -> int:
        if 'h' in duration_str:
            hours, mins = re.match(r"(\d+)h(\d+)", duration_str).groups()
            return int(hours)*60 + int(mins)
        return int(re.search(r"\d+", duration_str).group())

    @classmethod
    def get_location_score(cls, location: str) -> float:
        max_duration = 120
        for entry in TRANSPORT_DATA:
            if entry.get('station') == location or \
               entry.get('ville') == location or \
               entry.get('commune') == location:
                duration = cls.parse_duration(entry['temps_trajet'])
                return max(0, 1 - (duration / max_duration))
        return 0.7

class JobMatcher:
    INDUSTRY_WEIGHTS = {
        "PLM": 0.25, "Enovia": 0.2, "lean": 0.15,
        "six sigma": 0.15, "gestion de projet": 0.1
    }

    @classmethod
    def calculate_score(cls, job_desc: str, resume_text: str, location: str) -> float:
        try:
            if not job_desc or not resume_text:
                return 0.0
                
            embeddings = sbert_model.encode([job_desc, resume_text])
            base_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            boost = sum(weight for term, weight in cls.INDUSTRY_WEIGHTS.items()
                       if term in job_desc.lower())
            location_score = TransportAnalyzer.get_location_score(location)
            final_score = (0.7 * (base_score + boost)) + (0.3 * location_score)
            return round(max(0.0, min(final_score, 1.0)), 4)
        except Exception as e:
            print(f"⚠️ Score calculation error: {str(e)}")
            return 0.0

    @staticmethod
    def extract_key_skills(job_desc: str) -> List[str]:
        doc = nlp(job_desc.lower())
        return [term for term in JobMatcher.INDUSTRY_WEIGHTS.keys() 
               if term in job_desc.lower()]

class ResumeParser:
    @staticmethod
    def extract_text() -> str:
        with fitz.open(CONFIG["resume_path"]) as doc:
            return " ".join([page.get_text() for page in doc])

    @staticmethod
    def clean_text(text: str) -> str:
        replacements = {'\n•': ' ', '\n○': ' ', ' ': ' ', '\u2022': '•'}
        for k, v in replacements.items():
            text = text.replace(k, v)
        return re.sub(r'\s+', ' ', text).strip()

    @classmethod
    def analyze(cls) -> Dict:
        raw_text = cls.extract_text()
        if len(raw_text) < 300:
            raise ValueError("Insufficient text extracted from PDF")
        cleaned = cls.clean_text(raw_text)
        doc = nlp(cleaned)
        return {
            "competences": list({ent.text for ent in doc.ents 
                               if ent.label_ in ("SKILL", "PROFESSION")}),
            "full_text": cleaned
        }

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors='coerce', downcast='float'
    ).fillna(0.0)
    
    for col in df.select_dtypes(include=['object']):
        if len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    return df[[c for c in df.columns if not c.startswith('_raw')]].copy()

def search_jobs() -> pd.DataFrame:
    all_jobs = []
    seen_job_urls = set()  # Track unique job URLs across all searches
    resume_data = ResumeParser().analyze()
    
    # Pre-load existing jobs from database to avoid duplicates
    nocodb_client = NocoDBClient()
    existing_job_urls = nocodb_client.fetch_existing_job_urls()
    
    # Generate locations with null filtering
    locations = {
        val for e in TRANSPORT_DATA 
        for k in ['station', 'ville', 'commune'] 
        if (val := e.get(k)) is not None  # Filter out None values
    }
    
    print(f"🔍 Found {len(locations)} valid locations")
    print(f"📋 Already have {len(existing_job_urls)} jobs in database")

    for idx, location in enumerate(locations):
        if not location:  # Additional safety check
            continue

        print(f"\n📍 Processing location {idx+1}/{len(locations)}: {location}")
        
        for term in CONFIG["search_params"]["search_terms"]:
            for attempt in range(CONFIG["search_params"]["retry_attempts"]):
                try:
                    jobs = scrape_jobs(
                        site_name=["indeed", "linkedin"],
                        search_term=term,
                        location=location,
                        country_indeed='france',
                        results_wanted=CONFIG["search_params"]["max_results"],
                        hours_old=CONFIG["search_params"]["max_days_old"]*24,
                        linkedin_fetch_description=True
                    )
                    
                    if not jobs.empty:
                        # First, filter out job URLs we've already seen in this run
                        new_jobs = jobs[~jobs['job_url'].isin(seen_job_urls)]
                        
                        # Next, filter out jobs that already exist in the database
                        new_jobs = new_jobs[~new_jobs['job_url'].isin(existing_job_urls)]
                        
                        # Update our tracking set with these new URLs
                        seen_job_urls.update(new_jobs['job_url'].tolist())
                        
                        # Process in chunks for efficiency
                        if not new_jobs.empty:
                            chunks = [new_jobs.iloc[i:i+CONFIG["search_params"]["chunk_size"]] 
                                    for i in range(0, len(new_jobs), CONFIG["search_params"]["chunk_size"])]
                            
                            for chunk in chunks:
                                chunk['ai_score'] = chunk['description'].apply(
                                    lambda desc: JobMatcher.calculate_score(
                                        desc, resume_data["full_text"], location))
                                
                                filtered = chunk[chunk['ai_score'] >= CONFIG["ai"]["match_threshold"]]
                                
                                if not filtered.empty:
                                    all_jobs.append(filtered)
                            
                            print(f"✅ {term}: {len(new_jobs)}/{len(jobs)} new jobs (filtered {len(jobs) - len(new_jobs)} duplicates)")
                        else:
                            print(f"🟡 {term}: All {len(jobs)} jobs were duplicates")
                    break
                except Exception as e:
                    if attempt == CONFIG["search_params"]["retry_attempts"]-1:
                        print(f"🔴 Failed {term} in {location}: {str(e)}")
                    time.sleep(2 ** attempt)
    
    # Final consolidation and deduplication
    if all_jobs:
        combined_df = pd.concat(all_jobs, ignore_index=True)
        # One last deduplication for safety
        final_df = combined_df.drop_duplicates('job_url')
        print(f"🔄 Final deduplication: {len(combined_df)} → {len(final_df)} jobs")
        return final_df
    else:
        print("📭 No new unique jobs found")
        return pd.DataFrame()

class NocoDBClient:
    def __init__(self):
        self.base_url = CONFIG["nocodb"]["api_url"]
        self.token = CONFIG["nocodb"]["api_token"]
        self.table = CONFIG["nocodb"]["table_name"]
        self.headers = {"xc-token": self.token}
        self.expected_columns = CONFIG["nocodb"]["expected_columns"]
        self._existing_job_urls = None  # Cache for existing job URLs

    def prepare_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if 'id' not in dataframe.columns:
            dataframe = self._generate_ids(dataframe)
            
        df = dataframe.rename(columns={'id': 'id1'}) if 'id' in dataframe.columns else dataframe
        
        # Process relative dates first, then convert
        df = process_relative_dates(df)
        # Convert to datetime and format to YYYY-MM-DD (date only)
        df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
        # Store as string in YYYY-MM-DD format
        df['date_posted'] = df['date_posted'].dt.strftime('%Y-%m-%d')

        df['ai_score'] = df['ai_score'].fillna(0.0).round(4)
        df['min_amount'] = df['min_amount'].fillna(0.0)
        df['max_amount'] = df['max_amount'].fillna(0.0)
        
        return optimize_dataframe(df)[self.expected_columns]

    def _generate_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df['id'] = df['job_url'].str.extract(r'(\d+)$')
        
        mask = df['id'].isna()
        df.loc[mask, 'id'] = df[mask].index.astype(str) + '_' + \
                            df[mask]['company'].str[:3].fillna('job') + '_' + \
                            df[mask]['location'].str[:3].fillna('loc')
        
        return df
    
    def fetch_existing_job_urls(self) -> set:
        """Fetch all existing job URLs from the database to avoid duplicates"""
        if self._existing_job_urls is not None:
            return self._existing_job_urls
            
        try:
            print("📋 Fetching existing job URLs to avoid duplicates...")
            all_urls = set()
            page = 1
            page_size = 500
            
            while True:
                response = requests.get(
                    f"{self.base_url}/api/v2/tables/{self.table}/records",
                    params={"limit": page_size, "offset": (page-1) * page_size, "fields": "job_url"},
                    headers=self.headers,
                    timeout=15
                )
                response.raise_for_status()
                
                data = response.json()
                records = data.get('list', [])
                
                if not records:
                    break
                    
                for record in records:
                    if record.get('job_url'):
                        all_urls.add(record['job_url'])
                
                if len(records) < page_size:
                    break
                    
                page += 1
                
            print(f"✅ Found {len(all_urls)} existing job records")
            self._existing_job_urls = all_urls
            return all_urls
            
        except Exception as e:
            print(f"⚠️ Error fetching existing job URLs: {str(e)}")
            return set()

    def save_records(self, df: pd.DataFrame):
        if df.empty:
            print("🟡 No records to save")
            return
        
        # Get existing job URLs to avoid duplicates
        existing_urls = self.fetch_existing_job_urls()
        
        # Filter out duplicates
        new_df = df[~df['job_url'].isin(existing_urls)].copy()
        
        if new_df.empty:
            print("🟡 All jobs already exist in the database")
            return
            
        print(f"📊 Processing {len(new_df)}/{len(df)} new jobs (filtered {len(df) - len(new_df)} duplicates)")

        def clean_json_values(record):
            """Convert NaN, 'None' strings, and invalid values to proper JSON format"""
            cleaned_record = {}
            for key, value in record.items():
                if isinstance(value, str) and value.lower() == "none":
                    cleaned_record[key] = None  # Convert "None" string to null
                elif isinstance(value, float) and math.isnan(value):
                    cleaned_record[key] = None  # Convert NaN to null
                elif key in ["min_amount", "max_amount"] and value is None:
                    cleaned_record[key] = 0.0  # Set missing salary values to 0.0
                elif key == "date_posted":
                    if isinstance(value, (datetime, pd.Timestamp)):
                        # Format as YYYY-MM-DD
                        cleaned_record[key] = value.strftime('%Y-%m-%d')
                    elif isinstance(value, str):
                        if len(value) >= 10 and "-" in value:  # Already in ISO format or close
                            # Just keep the YYYY-MM-DD part
                            cleaned_record[key] = value[:10]
                        else:
                            try:
                                # Try to convert and format
                                cleaned_record[key] = pd.to_datetime(value).strftime('%Y-%m-%d')
                            except:
                                # Fallback to today's date if conversion fails
                                cleaned_record[key] = datetime.now().strftime('%Y-%m-%d')
                    else:
                        # Fallback to today's date for missing/null dates
                        cleaned_record[key] = datetime.now().strftime('%Y-%m-%d')
                elif key == "id":  
                    cleaned_record["id1"] = str(value)
                else:
                    cleaned_record[key] = value  
            return cleaned_record
        
        new_df = new_df.replace({np.nan: None, pd.NA: None, math.nan: None})

        for batch in np.array_split(new_df, max(1, len(new_df)//100)):
            records = batch.to_dict('records')
            success = 0
            for record in records:
                try:
                    record = clean_json_values(record)  # Apply cleanup
                    
                    if not record.get('job_url'):
                        continue
                        
                    # Skip if URL already exists (extra safety check)
                    if record.get('job_url') in existing_urls:
                        continue

                    # Validate AI score
                    record['ai_score'] = round(float(record.get('ai_score', 0)), 4)

                    # Send to NocoDB
                    response = requests.post(
                        f"{self.base_url}/api/v2/tables/{self.table}/records",
                        json=[record],
                        headers=self.headers,
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    # Update our cache of existing URLs
                    existing_urls.add(record.get('job_url'))
                    if self._existing_job_urls is not None:
                        self._existing_job_urls.add(record.get('job_url'))
                        
                    success += 1
                except Exception as e:
                    print(f"🔴 Failed to save {record.get('job_url', '')[:30]}")
                    print(f"Error details: {str(e)}")
                    print(f"Problematic record: {json.dumps(record, indent=2, default=str)[:200]}...")
            print(f"✅ Saved {success}/{len(batch)} records")
            time.sleep(2)

def daily_workflow():
    """Automated job search every 24 hours"""
    while True:
        try:
            print("\n=== Starting daily job search ===")
            jobs_df = search_jobs()
            
            if not jobs_df.empty:
                print(f"📊 Found {len(jobs_df)} new jobs")
                NocoDBClient().save_records(jobs_df)
            else:
                print("🛑 No new jobs found")
            
            time.sleep(86400)  # 24 hours
        except Exception as e:
            print(f"🔴 Daily workflow error: {str(e)}")
            time.sleep(3600)  # Retry after 1 hour

def validate_transport_data():
    invalid_entries = [
        e for e in TRANSPORT_DATA
        if not any(e.get(k) for k in ['station', 'ville', 'commune'])
    ]
    
    if invalid_entries:
        print(f"⚠️ Warning: Found {len(invalid_entries)} entries without location data")
        print("Problematic entries:", json.dumps(invalid_entries[:2], indent=2))

# ======================
# RUNNER
# ======================
if __name__ == "__main__":
    try:
        print("=== Job Search Pipeline ===")

        validate_transport_data()
        
        # Start background daily search
        threading.Thread(target=daily_workflow, daemon=True).start()
        
        # Initial immediate search
        jobs_df = search_jobs()
        
        if not jobs_df.empty:
            print(f"\n📊 Found {len(jobs_df)} jobs after filtering")
            NocoDBClient().save_records(jobs_df)
        else:
            print("\n🛑 No jobs found matching criteria")
            
    except Exception as e:
        print(f"\n🔴 Critical failure: {str(e)}")
        exit(1)
