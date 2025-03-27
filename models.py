from pydantic import BaseModel
from typing import List, Optional

class Skill(BaseModel):
    name: str

class JobListing(BaseModel):
    id: str
    site: str
    job_url: str
    title: str
    company: str 
    job_type: str
    date_posted: str
    description: str
    min_amount: Optional[float]  # Montant minimum
    max_amount: Optional[float]  # Montant maximum
    currency: str
    is_remote: bool
    job_level: str
    job_function: str

    # Ajoutez d'autres champs si nécessaire

class AnalysisResult(BaseModel):
    job_title: str
    strengths: List[str]
    weaknesses: List[str]
    matching_score: float 