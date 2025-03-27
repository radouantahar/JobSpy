# job_matching_system.py
import os
import fitz
import numpy as np
import spacy
import pandas as pd
import requests
import re
import time
import math
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from jobspy import scrape_jobs
from bs4 import BeautifulSoup
from collections import defaultdict
import ollama


# ======================
# CONFIGURATION
# ======================
CONFIG = {
    "resume_path": "RT_Resume_CV.pdf",
    "nocodb": {
        "api_url": "http://localhost:8080",
        "api_token": "MXnPaeY8ZY8pxPvuW6krLCgSGhQ_qRRlxH1vMbAc",
        "table_name": "ml9t4c3qql35wgb",
        "expected_columns": [
            "id1", "site", "job_url", "job_url_direct", "title", "company",
            "location", "date_posted", "job_type", "salary_source", "interval",
            "min_amount", "max_amount", "currency", "is_remote", "job_level",
            "job_function", "listing_type", "emails", "description",
            "company_industry", "company_url", "company_logo",
            "company_url_direct", "company_addresses", "company_num_employees",
            "company_revenue", "company_description", "ai_score", "Cum_Time_From_Reims"
        ]
    },
    "search_params": {
        "locations": ["Paris"],
        "search_terms": [
            "Chef de Projet Industrialisation",
            "Responsable Méthodes Industrielles",
            "Manager Amélioration Continue"
        ],
        "exact_match": True,
        # "required_keywords": ["industrialisation", "méthodes", "lean"],
        "max_results": 50,
        "max_days_old": 7,
        "retry_attempts": 3,
        "chunk_size": 100,
        "min_salary": 45000
    },
    "commute": {
        "home_address": "Reims, France",
        "max_commute": 90,
        "request_delay": 5,
        "retries": 3,
        "google_api_key": "AIzaSyBCqS5IOLNZ5wZaQfLB3Onqq6a9zdRT0Ac",
        "travel_modes": ["TRANSIT", "DRIVING"],
        "cache_ttl_hours": 24,
        "max_cache_size": 1000,
        "field_mask": "routes.duration"
    },
    "ai": {
        "spacy_model": "fr_core_news_sm",
        "sentence_model": "distiluse-base-multilingual-cased-v2",
        "industry_terms": {
            "industrialisation": {
                "weight": 0.3,  # Increased importance
                "synonyms": ["industrialization", "transfert industriel", "industrial project management"]
            },
            "chef de projet industrialisation": {
                "weight": 0.4,  # Highest weight for key role
                "synonyms": [
                    "responsable industrialisation",
                    "industrial project manager",
                    "PM industrialization",
                    "leader transfert technique",
                    "project lead manufacturing"
                ]
            },
            "gestion de projet": {
                "weight": 0.35,
                "synonyms": [
                    "project management", 
                    "PMO",
                    "chef de projet",
                    "project lead",
                    "Pilote projet",
                    "PM industriel",
                    "project governance"
                ]
            },
            "PLM": {
                "weight": 0.35,  # Core technical skill
                "synonyms": ["Enovia V6", "product lifecycle management", "digital mock-up", "EBOM management"]
            },
            "lean manufacturing": {
                "weight": 0.25,
                "synonyms": ["lean", "TPS", "Toyota Production System", "value stream"]
            },
            "six sigma": {
                "weight": 0.2,
                "synonyms": ["6 sigma", "green belt", "DMAIC", "process variation control"]
            },
            "amélioration continue": {
                "weight": 0.3,
                "synonyms": ["kaizen", "continuous improvement", "PDCA", "process optimization", "CIP"]
            },
            "BPMN2": {
                "weight": 0.2,
                "synonyms": ["business process modeling", "process mapping", "JIRA workflow"]
            },
            "Pilote implantation": {
                "weight": 0.25,
                "synonyms": [
                    "factory layout manager",
                    "plant relocation lead",
                    "implantation industrielle",
                    "production floor optimization",
                    "value stream design"
                ]
            },
            "robotique": {
                "weight": 0.2,
                "synonyms": ["robotics", "automatisation", "cobotique", "automation cell"]
            },
            "digitalisation": {
                "weight": 0.25,
                "synonyms": ["digital transformation", "industrie 4.0", "IIoT", "smart factory"]
            },
            "analyse de la valeur": {
                "weight": 0.2,
                "synonyms": ["value stream mapping", "VSM", "process analysis", "time-motion study"]
            },
            "gestion des risques": {
                "weight": 0.2,
                "synonyms": ["risk management", "FMEA", "APQP", "non-conformity control"]
            },
            "soudure robotique": {
                "weight": 0.15,
                "synonyms": ["robotic welding", "automated joining", "spot welding automation"]
            },
            "JIRA Agile SAFe": {
                "weight": 0.25,
                "synonyms": ["Scaled Agile", "Agile framework", "sprint planning", "PI planning"]
            },
            "QCD": {
                "weight": 0.2,
                "synonyms": ["qualité-coût-délai", "quality-cost-delivery", "production triangle"]
            }
        },
        "score_weights": {
            "semantic": 0.4,
            "keywords": 0.5,
            "salary": 0.1
        },
        "match_threshold": 0.45
    }
}



OLLAMA_CONFIG = {
    "model": "llama3.2",  # Using 8B parameter version
    "base_url": "http://localhost:11434",
    "cv_keywords_prompt": """
    Analyze this resume text and extract technical skills and industry-specific terms. 
    Focus on manufacturing, engineering, and project management terminology.
    Return only a JSON array of keywords in French:
    {
        "competences": ["keyword1", "keyword2", ...]
    }
    """
}

# Initialize NLP components
def enhance_ner():
    nlp = spacy.load(CONFIG["ai"]["spacy_model"])

    industrial_terms = [
    "industrialisation", "lean manufacturing", "six sigma",
    "amélioration continue", "gestion de projet", "PLM",
    "Enovia", "BPMN2", "robotique", "automatisation"
    ]

    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "SKILL", "pattern": term} for term in industrial_terms]
    ruler.add_patterns(patterns)

    return nlp

nlp = enhance_ner() 

# ======================
# CORE COMPONENTS
# ======================
class OllamaResumeParser:
    @staticmethod
    def extract_text() -> str:
        """PDF text extraction with debug capabilities"""
        try:
            text = []
            print("\n=== RESUME PARSING DEBUG ===")
            print(f"Attempting to parse: {CONFIG['resume_path']}")
            
            with fitz.open(CONFIG["resume_path"]) as doc:
                print(f"Number of pages: {len(doc)}")
                
                for page_num, page in enumerate(doc):
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    
                    page_text = "\n".join([b[4].strip() for b in blocks if b[4].strip()])
                    text.append(page_text)
                    
                    # Debug: show first block of each page
                    print(f"\nPage {page_num + 1} first content block:")
                    print(blocks[0][4][:100] + "..." if blocks else "Empty page")

            full_text = "\n".join(text)
            print("\n=== RAW TEXT EXTRACT ===")
            print(f"Character count: {len(full_text)}")
            print("Sample text (first 500 chars):")
            print(full_text[:500] + ("..." if len(full_text) > 500 else ""))
            
            return full_text
            
        except Exception as e:
            print(f"\n⚠️ Resume parsing failed: {str(e)}")
            return ""

    @classmethod
    def analyze(cls) -> Dict:
        raw_text = cls.extract_text()
        
        try:
            # Call Ollama API
            response = ollama.generate(
                model=OLLAMA_CONFIG["model"],
                prompt=f"{OLLAMA_CONFIG['cv_keywords_prompt']}\n{raw_text}",
                format="json",
                options={"temperature": 0.1}
            )
            
            # Parse response
            keywords = json.loads(response['response'])['competences']
            print("Llama3-extracted competences:", keywords)
            
            return {
                "competences": keywords,
                "full_text": raw_text
            }
            
        except Exception as e:
            print(f"⚠️ Ollama error: {str(e)}")
            return {"competences": [], "full_text": raw_text}
        
class SemanticMatcher:
    def __init__(self):
        self.resume_data = OllamaResumeParser.analyze()
        self.model = SentenceTransformer(CONFIG["ai"]["sentence_model"])
        self.industry_terms = CONFIG["ai"]["industry_terms"]

    def calculate_score(self, job_desc: str) -> float:
        # Calculate keyword score using industry terms
        keyword_score = 0.0
        job_desc_lower = job_desc.lower()
        
        for term, data in self.industry_terms.items():
            search_terms = [term] + data.get("synonyms", [])
            for t in search_terms:
                if t.lower() in job_desc_lower:
                    keyword_score += data["weight"]
                    break  # Avoid duplicate matches

        # Calculate semantic similarity
        embeddings = self.model.encode([job_desc, self.resume_data["full_text"]])
        semantic_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Combine scores using configured weights
        weights = CONFIG["ai"]["score_weights"]
        combined_score = (
            (semantic_score * weights["semantic"]) +
            (keyword_score * weights["keywords"])
        )
        
        return min(max(combined_score, 0.0), 1.0)  # Ensure score between 0-1        

class CommuteCalculator:
    def __init__(self):
        self.cache = CommuteCache()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'fr-FR,fr;q=0.9'
        })

    def get_commute_time(self, location: str) -> int:
        normalized_loc = self.normalize_location(location)
        if cached := self.cache.get(normalized_loc):
            return cached

        for mode in CONFIG["commute"]["travel_modes"]:
            try:
                duration = self._calculate_route(normalized_loc, mode)
                if duration > 0:
                    self.cache.set(normalized_loc, duration)
                    return duration
            except Exception as e:
                print(f"⚠️ Commute error ({mode}): {str(e)}")
                continue
        return -1

    def _calculate_route(self, location: str, mode: str) -> int:
        endpoint = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": CONFIG["commute"]["google_api_key"],
            "X-Goog-FieldMask": CONFIG["commute"]["field_mask"]
        }

        body = {
            "origin": {"address": CONFIG["commute"]["home_address"]},
            "destination": {"address": location},
            "travelMode": mode,
            "departureTime": datetime.now(timezone.utc).isoformat(),
            "computeAlternativeRoutes": False,
            "languageCode": "fr-FR",
            "units": "METRIC"
        }

        response = self.session.post(endpoint, json=body, headers=headers, timeout=15)
        response.raise_for_status()
        return int(math.ceil(float(response.json()["routes"][0]["duration"][:-1]) / 60))

    @staticmethod
    def normalize_location(location: str) -> str:
        return re.sub(r'\s+', ' ', location.lower().strip().replace("île-de-france", ""))

# ======================
# DATA MANAGEMENT
# ======================
class CommuteCache:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
        self.load_cache()
        
    def load_cache(self):
        try:
            with open('commute_cache.json', 'r') as f:
                self.cache = json.load(f)
            print(f"✅ Loaded {len(self.cache)} cached locations")
        except FileNotFoundError:
            self.cache = {}
            
    def save_cache(self):
        with self.lock:
            with open('commute_cache.json', 'w') as f:
                json.dump(self.cache, f)
                
    def get(self, location: str) -> Optional[int]:
        key = CommuteCalculator.normalize_location(location)
        with self.lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
            
    def set(self, location: str, minutes: int):
        key = CommuteCalculator.normalize_location(location)
        with self.lock:
            if len(self.cache) >= CONFIG["commute"]["max_cache_size"]:
                self.cache.popitem()
            self.cache[key] = minutes
            self.save_cache()

class JobDeduplicator:
    def __init__(self, nocodb_client):
        self.nocodb_client = nocodb_client
        self.existing_urls: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.bloom_filter = bytearray(1000000)
        self.bloom_size = len(self.bloom_filter) * 8
        self.lock = threading.Lock()

    def initialize(self):
        print("🔍 Loading existing job URLs...")
        page = 1
        batch_size = 500
        
        while True:
            with self.lock:
                urls = self.nocodb_client.fetch_url_batch(page, batch_size)
                if not urls:
                    break
                self._add_to_bloom(urls)
                self.existing_urls.update(urls)
                page += 1
                time.sleep(0.2)

    def _add_to_bloom(self, urls: Set[str]):
        for url in urls:
            for hash_val in self._get_hashes(url):
                self.bloom_filter[hash_val // 8] |= 1 << (hash_val % 8)
                
    def _get_hashes(self, url: str) -> List[int]:
        return [(hash(url) + i) % self.bloom_size for i in range(3)]
        
    def is_duplicate(self, url: str) -> bool:
        normalized = URLNormalizer.normalize(url)
        if not normalized:
            return True
            
        bloom_match = all(
            self.bloom_filter[hash_val // 8] & (1 << (hash_val % 8))
            for hash_val in self._get_hashes(normalized)
        )
        
        with self.lock:
            return normalized in self.existing_urls or normalized in self.seen_urls
            
    def add_urls(self, urls: List[str]):
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

    def fetch_url_batch(self, page: int, batch_size: int) -> Set[str]:
        try:
            response = requests.get(
                f"{self.base_url}/api/v2/tables/{self.table}/records",
                params={"fields": "job_url", "limit": batch_size, "offset": (page-1)*batch_size},
                headers=self.headers,
                timeout=15
            )
            return {
                URLNormalizer.normalize(r.get('job_url', ''))
                for r in response.json().get('list', [])
                if r.get('job_url')
            }
        except Exception as e:
            print(f"⚠️ Batch fetch error: {str(e)}")
            return set()

    def save_records(self, df: pd.DataFrame, deduplicator: JobDeduplicator):
        if df.empty:
            return

        df = self.prepare_data(df)
        records = df.applymap(lambda x: None if pd.isna(x) else x).to_dict('records')
        
        BATCH_SIZE = 50
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            try:
                response = requests.post(
                    f"{self.base_url}/api/v2/tables/{self.table}/records/bulk",
                    json=batch,
                    headers=self.headers,
                    timeout=20
                )
                response.raise_for_status()
                deduplicator.add_urls([r['job_url'] for r in batch])
            except Exception as e:
                print(f"🔴 Batch insert failed: {str(e)}")

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = process_relative_dates(df)
        df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce', utc=True)
        return optimize_dataframe(df)[CONFIG["nocodb"]["expected_columns"]]

class URLNormalizer:
    TRACKING_PARAMS = {'utm_', 'ref', 'source', 'lr', 'from', 'fbclid', 'campaign'}
    
    @classmethod
    def normalize(cls, url: str) -> str:
        try:
            parsed = urlparse(url)
            filtered_query = "&".join(
                q for q in parsed.query.split("&")
                if not any(q.startswith(p) for p in cls.TRACKING_PARAMS))
            path = parsed.path.rstrip('/').lower() or '/'
            
            return urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                path,
                parsed.params,
                filtered_query,
                ''  
            ))
        except:
            return ""

# ======================
# UTILITIES
# ======================
def process_relative_dates(df: pd.DataFrame) -> pd.DataFrame:
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    def parse_date(date_str):
        if not isinstance(date_str, str):
            return date_str
            
        date_str = date_str.lower().strip()
        matches = re.findall(r'(\d+)', date_str)
        
        if 'hour' in date_str or 'hr' in date_str:
            hours = int(matches[0]) if matches else 1
            return today - timedelta(hours=hours)
        elif 'day' in date_str:
            days = int(matches[0]) if matches else 1
            return today - timedelta(days=days)
        elif 'week' in date_str:
            weeks = int(matches[0]) if matches else 1
            return today - timedelta(weeks=weeks)
        elif 'month' in date_str:
            months = int(matches[0]) if matches else 1
            return today - timedelta(days=30*months)
        return date_str
    
    if 'date_posted' in df.columns:
        df['date_posted'] = df['date_posted'].apply(parse_date)
    return df

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['object']):
        if len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    
    return df

def apply_commute_penalty(score: float, commute_time: int) -> float:
    if commute_time == -1:
        return score
    penalty = 1 / (1 + math.exp(-0.1 * (commute_time - 45))) * 0.3
    return max(0, score * (1 - penalty))

class SalaryExtractor:
    @staticmethod
    def extract(description: str) -> Tuple[Optional[int], Optional[int]]:
        patterns = [
            r'(?:€|euro)\s*(\d{1,3}(?:[,\s]\d{3})+)',  # European format
            r'\b(\d+)\s*k\s*€?\b',  # 50k format
            r'(?:de|jusqu\'à)\s+(\d+)\s*€'  # French phrases
        ]
        
        max_sal = 0
        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if 'k' in match:
                    val = int(match.replace('k', '')) * 1000
                else:
                    val = int(match.replace(' ', '').replace(',', ''))
                max_sal = max(max_sal, val)
        return (None, max_sal) if max_sal > 0 else (None, None)

# ======================
# MAIN PIPELINE
# ======================
def main():
    
    # Add temporary test
    test_desc = "Salaire: 55 000€ à 65 000€ par an + avantages"
    print("\nSalary test:", SalaryExtractor.extract(test_desc))

    print("=== JOB MATCHING PIPELINE ===")
    start_time = time.time()
    
    # Initialize components
    deduplicator = JobDeduplicator(NocoDBClient())
    deduplicator.initialize()
    
    matcher = SemanticMatcher()
    commute_calculator = CommuteCalculator()
    
    # Scrape jobs with retry
    jobs = pd.DataFrame()
    for search_term in CONFIG["search_params"]["search_terms"]:
        for location in CONFIG["search_params"]["locations"]:
            result = scrape_jobs(
                site_name=["indeed", "linkedin"],
                search_term=search_term,  # Single string
                location=location,  # Single string
                country_indeed='france',
                results_wanted=CONFIG["search_params"]["max_results"],
                hours_old=CONFIG["search_params"]["max_days_old"]*24,
                linkedin_fetch_description=True,
                pages="1"  # Start with first page
            )
            jobs = pd.concat([jobs, result], ignore_index=True)
    
    # Filter duplicates
    df = jobs.copy()
    df['normalized_url'] = df['job_url'].apply(URLNormalizer.normalize)
    df['is_duplicate'] = df['normalized_url'].apply(deduplicator.is_duplicate)
    filtered = df[~df['is_duplicate']].drop(columns=['normalized_url', 'is_duplicate'])
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, chunk in filtered.groupby(np.arange(len(filtered)) // CONFIG["search_params"]["chunk_size"]):
            futures.append(executor.submit(
                process_chunk,
                chunk,
                matcher,
                commute_calculator,
                deduplicator
            ))
        
        results = [f.result() for f in as_completed(futures) if not f.result().empty]
    
    # Save results
    final_df = pd.concat(results) if results else pd.DataFrame()
    NocoDBClient().save_records(final_df, deduplicator)

    print("\n=== RAW SCRAPED JOBS ===")
    print(jobs[['title', 'company', 'location']].head(10))
    
    print(f"\n✅ Process completed in {time.time()-start_time:.2f}s")
    print(f"Results: {len(final_df)} matching jobs")
    print(f"Cache stats: Hits={commute_calculator.cache.hits} Misses={commute_calculator.cache.misses}")
    
    if not final_df.empty:
        print(final_df[['title', 'company', 'ai_score', 'Cum_Time_From_Reims']].head())

def process_chunk(chunk: pd.DataFrame, matcher: SemanticMatcher,
                 commute_calculator: CommuteCalculator, deduplicator: JobDeduplicator) -> pd.DataFrame:
    chunk = chunk.copy()
    
    # Calculate initial scores
    chunk['ai_score'] = chunk['description'].apply(matcher.calculate_score)
    
    # Debug: Show jobs before any filtering
    print("\n=== BEFORE FILTERING ===")
    print(f"Total jobs in chunk: {len(chunk)}")
    print(chunk[['title', 'company', 'location', 'ai_score']].head())
    
    # Apply commute calculation
    threshold_mask = chunk['ai_score'] >= CONFIG["ai"]["match_threshold"]
    chunk.loc[threshold_mask, 'Cum_Time_From_Reims'] = chunk.loc[threshold_mask, 'location'].apply(
        lambda loc: commute_calculator.get_commute_time(str(loc)))
    
    # Apply score penalty
    chunk['ai_score'] = chunk.apply(
        lambda row: apply_commute_penalty(row['ai_score'], row['Cum_Time_From_Reims']),
        axis=1
    )
    
    # Debug: Show jobs after score calculation but before final filtering
    print("\n=== AFTER SCORE CALCULATION ===")
    print(chunk[['title', 'company', 'ai_score', 'Cum_Time_From_Reims']].head())
    
    # Apply final filters
    filtered = chunk[
        (chunk['ai_score'] >= CONFIG["ai"]["match_threshold"]) &
        ((chunk['Cum_Time_From_Reims'] <= CONFIG["commute"]["max_commute"]) | 
         (chunk['Cum_Time_From_Reims'] == -1))
    ]
    
    # Debug: Show final filtered results
    print("\n=== FINAL FILTERED JOBS ===")
    print(f"Remaining jobs: {len(filtered)}")
    if not filtered.empty:
        print(filtered[['title', 'company', 'ai_score', 'Cum_Time_From_Reims']])
    else:
        print("No jobs passed all filters")
    
    return filtered

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n🔴 Critical failure: {str(e)}")
        exit(1)