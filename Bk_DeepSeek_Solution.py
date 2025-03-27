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
from typing import Dict, List, Set
import threading
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            "company_revenue", "company_description", "ai_score"
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

nlp = spacy.load(CONFIG["ai"]["spacy_model"])
sbert_model = SentenceTransformer(CONFIG["ai"]["sentence_model"])

# tps_transport_L_Test

with open('tps_transport_L.json', 'r', encoding='utf-8') as f:
    TRANSPORT_DATA = json.load(f)

# ======================
# CORE COMPONENTS
# ======================
class URLNormalizer:
    TRACKING_PARAMS = {'utm_', 'ref', 'source', 'lr', 'from', 'fbclid', 'campaign'}
    
    @classmethod
    def normalize(cls, url: str) -> str:
        """Advanced URL normalization with parameter filtering and path standardization"""
        if not isinstance(url, str) or not url.startswith("http"):
            return ""
            
        try:
            parsed = urlparse(url)
            
            # Filter query parameters
            filtered_query = "&".join(
                q for q in parsed.query.split("&")
                if not any(q.startswith(p) for p in cls.TRACKING_PARAMS)
            )
            
            # Standardize path
            path = parsed.path.rstrip('/').lower() or '/'
            
            return urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                path,
                parsed.params,
                filtered_query,
                ''  # Remove fragment
            ))
        except:
            return ""

class JobDeduplicator:
    def __init__(self, nocodb_client):
        self.nocodb_client = nocodb_client
        self.existing_urls: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.bloom_filter = bytearray(1000000)  # 1MB Bloom filter
        self.bloom_size = len(self.bloom_filter) * 8  # Calculate in bits
        self.lock = threading.Lock()

    def initialize(self):
        """Initialize with existing URLs using batch processing"""
        print("🔍 Loading existing job URLs...")
        start_time = time.time()
        page = 1
        batch_size = 500  # Reduced batch size for reliability
        
        try:
            while True:
                with self.lock:
                    urls = self.nocodb_client.fetch_url_batch(page, batch_size)
                    if not urls:
                        break
                    self._add_to_bloom(urls)
                    self.existing_urls.update(urls)
                    page += 1
                    time.sleep(0.2)  # Rate limiting

            print(f"✅ Loaded {len(self.existing_urls)} URLs in {time.time()-start_time:.2f}s")
        except Exception as e:
            print(f"🔴 Initialization failed: {str(e)}")
            raise

    def _add_to_bloom(self, urls: Set[str]):
        """Add URLs to Bloom filter"""
        for url in urls:
            for hash_val in self._get_hashes(url):
                self.bloom_filter[hash_val // 8] |= 1 << (hash_val % 8)
                
    def _get_hashes(self, url: str) -> List[int]:
        """Generate multiple hashes for Bloom filter"""
        return [
            (hash(url) + i) % self.bloom_size
            for i in range(3)
        ]
        
    def is_duplicate(self, url: str) -> bool:
        """Efficient duplicate check with Bloom filter optimization"""
        normalized = URLNormalizer.normalize(url)
        if not normalized:
            return True
            
        # Bloom filter check
        bloom_match = all(
            self.bloom_filter[hash_val // 8] & (1 << (hash_val % 8))
            for hash_val in self._get_hashes(normalized)
        )
        
        if not bloom_match:
            return False
            
        # Full check with thread-safe access
        with self.lock:
            return normalized in self.existing_urls or normalized in self.seen_urls
            
    def add_urls(self, urls: List[str]):
        """Add new URLs to tracking systems"""
        with self.lock:
            normalized_urls = {URLNormalizer.normalize(u) for u in urls}
            self.seen_urls.update(normalized_urls)
            self._add_to_bloom(normalized_urls)

class NocoDBClient:
    def __init__(self):
        self.base_url = CONFIG["nocodb"]["api_url"]
        self.token = CONFIG["nocodb"]["api_token"]
        self.table = CONFIG["nocodb"]["table_name"]
        self.headers = {"xc-token": self.token}
        self.expected_columns = CONFIG["nocodb"]["expected_columns"]
        
        # Verify connection and table existence on initialization
        self._validate_connection()

    def _validate_connection(self):
        """Validate connection to NocoDB and table existence"""
        try:
            # Test the connection with a simple API call to get table info
            response = requests.get(
                f"{self.base_url}/api/v1/db/meta/tables/{self.table}/exports",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 401:
                print(f"🔴 Authentication failed - check your API token")
                # Don't raise here to allow retries with different auth
            elif response.status_code == 404:
                print(f"🔴 Table '{self.table}' not found - verify table name")
                print(f"🔶 Will attempt to create records using single record API instead of bulk")
                # Don't raise here to allow fallback to single record insertion
            else:
                response.raise_for_status()
                print(f"✅ Successfully connected to NocoDB - Table '{self.table}' exists")
                
        except requests.ConnectionError:
            print(f"🔴 Cannot connect to NocoDB at {self.base_url} - check if server is running")
            # Don't raise to allow retries later if server starts
        except Exception as e:
            print(f"⚠️ Connection validation warning: {str(e)}")
            # Continue anyway to allow retries

    def _generate_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate unique IDs for jobs"""
        df = df.copy()
        df['id'] = df['job_url'].str.extract(r'(\d+)$').fillna(
            df.index.astype(str) + '_' + 
            df['company'].str[:3].fillna('job') + '_' +
            df['location'].str[:3].fillna('loc')
        )
        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data preparation with enhanced type handling"""
        df = df.copy()
        
        # ID generation
        if 'id' not in df.columns:
            df = self._generate_ids(df)
        
        # Rename id to id1 if needed
        df = df.rename(columns={'id': 'id1'}) if 'id' in df.columns else df
        
        # Date handling
        df = process_relative_dates(df)
        df['date_posted'] = pd.to_datetime(
            df['date_posted'], errors='coerce', utc=True
        ).dt.strftime('%Y-%m-%d')
        
        # Score normalization
        if 'ai_score' in df.columns:
            df['ai_score'] = df['ai_score'].fillna(0.0).clip(0, 1).round(4)
        
        # Fill numeric values - avoid FutureWarning
        if 'min_amount' in df.columns:
            df['min_amount'] = pd.to_numeric(df['min_amount'], errors='coerce').fillna(0.0)
        else:
            df['min_amount'] = 0.0
            
        if 'max_amount' in df.columns:
            df['max_amount'] = pd.to_numeric(df['max_amount'], errors='coerce').fillna(0.0)
        else:
            df['max_amount'] = 0.0
        
        return optimize_dataframe(df)[self.expected_columns]

    def fetch_url_batch(self, page: int, batch_size: int) -> Set[str]:
        """Safe batch URL fetching with error handling"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v2/tables/{self.table}/records",
                params={
                    "fields": "job_url",
                    "limit": batch_size,
                    "offset": (page-1)*batch_size,
                },
                headers=self.headers,
                timeout=15
            )
            
            # Handle 404 error gracefully
            if response.status_code == 404:
                print(f"⚠️ Table '{self.table}' not found during fetch_url_batch")
                return set()
                
            response.raise_for_status()
            
            return {
                URLNormalizer.normalize(r.get('job_url', ''))
                for r in response.json().get('list', [])
                if r.get('job_url')
            }
        except requests.HTTPError as e:
            print(f"⚠️ HTTP error {e.response.status_code}: {str(e)}")
            return set()
        except Exception as e:
            print(f"⚠️ Batch fetch error: {str(e)}")
            return set()

    def save_records(self, df: pd.DataFrame, deduplicator: JobDeduplicator):
        """Optimized save with bulk insert support, single record fallback, and proper NaN handling"""
        if df.empty:
            return

        # Prepare data
        df = self.prepare_data(df)
        
        # Replace NaN values with None for JSON compatibility
        records = df.replace({np.nan: None, pd.NA: None}).to_dict('records')
        
        # Additional NaN handling for nested structures and special cases
        for record in records:
            for key, value in record.items():
                # Check if value is float and NaN (catches np.nan and float('nan'))
                if isinstance(value, float) and math.isnan(value):
                    record[key] = None
        
        # Bulk insert in batches
        BATCH_SIZE = 50
        success_count = 0
        bulk_api_working = True  # Flag to track if bulk API works
        
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            
            # Try bulk insert first (if still considered working)
            if bulk_api_working:
                try:
                    response = requests.post(
                        f"{self.base_url}/api/v2/tables/{self.table}/records/bulk",
                        json=batch,
                        headers=self.headers,
                        timeout=20
                    )
                    
                    if response.status_code == 404:
                        print(f"🔶 Bulk API endpoint not found, falling back to single record insertion")
                        bulk_api_working = False
                        # Don't count as success yet, will try individual inserts
                    else:
                        response.raise_for_status()
                        success_count += len(batch)
                        
                        # Update deduplicator
                        deduplicator.add_urls([r['job_url'] for r in batch])
                        continue  # Skip to next batch
                        
                except Exception as e:
                    print(f"🔶 Bulk insert failed: {str(e)}")
                    print(f"🔶 Falling back to single record insertion")
                    bulk_api_working = False
                    # Fall through to individual inserts
            
            # Fallback to individual inserts if bulk failed
            if not bulk_api_working:
                for record in batch:
                    try:
                        # Try v2 API endpoint first
                        response = requests.post(
                            f"{self.base_url}/api/v2/tables/{self.table}/records",
                            json=record,
                            headers=self.headers,
                            timeout=10
                        )
                        
                        # If v2 fails with 404, try v1 endpoint
                        if response.status_code == 404:
                            response = requests.post(
                                f"{self.base_url}/api/v1/db/data/noco/{self.table}/views/{self.table}",
                                json=record,
                                headers=self.headers,
                                timeout=10
                            )
                            
                        response.raise_for_status()
                        success_count += 1
                        
                        # Update deduplicator
                        deduplicator.add_urls([record['job_url']])
                        
                    except Exception as e:
                        print(f"🔴 Single insert failed for record {record.get('id1', 'unknown')}: {str(e)}")
        
        print(f"✅ Saved {success_count}/{len(records)} records")

    # ======================
    # JOB PROCESSING
    # ======================

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
        "Chef de projet": 0.25, "industrialisation": 0.2, "lean": 0.15,
        "six sigma": 0.15, "gestion de projet": 0.2, "developpement produit": 0.1, "amelioration continue": 0.1, "PMO": 0.2, "PM": 0.2
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

# ======================
# UTILITIES
# ======================

def search_jobs() -> pd.DataFrame:
    """Optimized job search with parallel processing"""
    nocodb_client = NocoDBClient()
    deduplicator = JobDeduplicator(nocodb_client)
    deduplicator.initialize()
    
    resume_data = ResumeParser().analyze()
    locations = get_valid_locations()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for location in locations:
            for term in CONFIG["search_params"]["search_terms"]:
                futures.append(executor.submit(
                    process_location_term,
                    location, term, resume_data, deduplicator
                ))
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"🔴 Processing failed: {str(e)}")
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def process_location_term(location: str, term: str, resume_data: dict, deduplicator: JobDeduplicator) -> pd.DataFrame:
    """Process jobs for a specific location and search term"""
    print(f"\n📍 Processing {term} in {location}")
    all_jobs = []
    
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
            
            if jobs.empty:
                return pd.DataFrame()
            
            # Parallel processing of job chunks
            with ThreadPoolExecutor() as chunk_executor:
                chunk_futures = [
                    chunk_executor.submit(
                        process_job_chunk,
                        chunk,
                        resume_data,
                        location,
                        deduplicator
                    )
                    for chunk in split_dataframe(jobs, CONFIG["search_params"]["chunk_size"])
                ]
                
                for future in as_completed(chunk_futures):
                    result = future.result()
                    if not result.empty:
                        all_jobs.append(result)
            
            break  # Success - exit retry loop
        except Exception as e:
            if attempt == CONFIG["search_params"]["retry_attempts"]-1:
                print(f"🔴 Failed {term} in {location}: {str(e)}")
            time.sleep(2 ** attempt)
    
    return pd.concat(all_jobs, ignore_index=True) if all_jobs else pd.DataFrame()

def process_job_chunk(chunk: pd.DataFrame, resume_data: dict, location: str, deduplicator: JobDeduplicator) -> pd.DataFrame:
    """Process a chunk of jobs with deduplication and scoring"""
    # Create a copy to avoid the SettingWithCopyWarning
    chunk_copy = chunk.copy()
    
    # Apply duplicate filter
    chunk_copy.loc[:, 'is_duplicate'] = chunk_copy['job_url'].apply(deduplicator.is_duplicate)
    filtered = chunk_copy[~chunk_copy['is_duplicate']].copy()
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Score remaining jobs
    filtered.loc[:, 'ai_score'] = filtered['description'].apply(
        lambda desc: JobMatcher.calculate_score(desc, resume_data["full_text"], location))
    
    # Add to deduplicator before final filtering
    deduplicator.add_urls(filtered['job_url'].tolist())
    
    # Apply score threshold
    return filtered[filtered['ai_score'] >= CONFIG["ai"]["match_threshold"]]

def split_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

def get_valid_locations() -> Set[str]:
    """Get valid locations from transport data"""
    return {
        val for e in TRANSPORT_DATA 
        for k in ['station', 'ville', 'commune'] 
        if (val := e.get(k)) is not None
    }

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe memory usage"""
    df = df.copy()
    # Numeric columns
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')
    
    # Categorical columns
    for col in df.select_dtypes(include='object'):
        if len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    
    return df

def validate_transport_data():
    invalid_entries = [
        e for e in TRANSPORT_DATA
        if not any(e.get(k) for k in ['station', 'ville', 'commune'])
    ]
    
    if invalid_entries:
        print(f"⚠️ Warning: Found {len(invalid_entries)} entries without location data")
        print("Problematic entries:", json.dumps(invalid_entries[:2], indent=2))

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


# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    try:
        print("=== Job Search Pipeline ===")
        validate_transport_data()
        
        # Initialize NocoDBClient and test connection explicitly
        nocodb_client = NocoDBClient()
        print("Testing NocoDB connection...")
        
        # Initialize deduplicator only after confirming connection
        deduplicator = JobDeduplicator(nocodb_client)
        deduplicator.initialize()
          
        # Initial immediate search
        jobs_df = search_jobs()
        
        if not jobs_df.empty:
            print(f"\n📊 Found {len(jobs_df)} jobs after filtering")
            nocodb_client.save_records(jobs_df, deduplicator)
        else:
            print("\n🛑 No jobs found matching criteria")
            
    except Exception as e:
        print(f"\n🔴 Critical failure: {str(e)}")
        exit(1)