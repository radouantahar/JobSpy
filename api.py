# api.py
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Dict, Optional
import requests
import json
import os
from datetime import datetime, timezone
from DeepSeek_Solution import (
    ResumeParser,
    CONFIG,
    TransportAnalyzer,
    JobMatcher,
    optimize_dataframe,
    TRANSPORT_DATA
)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# API COMPONENTS
# ======================
class LocalLLM:
    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{CONFIG['ai']['local_llm']['base_url']}/api/generate",
                json={
                    "model": CONFIG["ai"]["local_llm"]["model"],
                    "prompt": prompt,
                    "temperature": CONFIG["ai"]["local_llm"]["temperature"],
                    "max_tokens": CONFIG["ai"]["local_llm"]["max_tokens"]
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"LLM Error: {str(e)}")

class DocumentGenerator:
    def __init__(self):
        self.llm = LocalLLM()
        self.resume_parser = ResumeParser()
        self.resume_data = self.resume_parser.analyze()

    def generate_cover_letter(self, job_data: Dict) -> str:
        """Generate tailored cover letter with SWOT insights"""
        try:
            swot = self._perform_swot_analysis(job_data)
            prompt = self._build_cover_prompt(job_data, swot)
            return self.llm.generate(prompt)
        except Exception as e:
            raise RuntimeError(f"Cover letter generation failed: {str(e)}")

    def tailor_resume(self, job_data: Dict) -> str:
        """Generate tailored resume based on job requirements"""
        try:
            prompt = self._build_resume_prompt(job_data)
            return self.llm.generate(prompt)
        except Exception as e:
            raise RuntimeError(f"Resume tailoring failed: {str(e)}")

    def _perform_swot_analysis(self, job_data: Dict) -> Dict:
        """Enhanced SWOT analysis with transport scoring"""
        location_score = TransportAnalyzer.get_location_score(job_data.get('location', ''))
        key_skills = JobMatcher.extract_key_skills(job_data.get('description', ''))
        
        prompt = f"""<s>[INST] <<SYS>>
        Perform SWOT analysis comparing resume to job:
        - Resume: {self.resume_data['full_text'][:3000]}
        - Job: {job_data.get('description', '')[:2000]}
        - Required Skills: {key_skills}
        - Company: {job_data.get('company_description', '')[:1000]}
        - Location Score: {location_score}/1
        
        Format as JSON: {{"strengths": [], "weaknesses": [], 
        "opportunities": [], "threats": []}}<</SYS>>[/INST]"""
        
        response = self.llm.generate(prompt)
        return json.loads(response)

    def _build_cover_prompt(self, job_data: Dict, swot: Dict) -> str:
        """Construct cover letter prompt with context"""
        transport_time = next((e['temps_trajet'] for e in TRANSPORT_DATA 
                             if e.get('ville') == job_data.get('location')), "N/A")
        
        return f"""<s>[INST] <<SYS>>
        Write professional French cover letter:
        - Address: {job_data.get('company', 'Hiring Manager')}
        - Position: {job_data.get('title', 'position')}
        - Key strengths: {swot['strengths'][:3]}
        - Address weakness: {swot['weaknesses'][0] if swot['weaknesses'] else ''}
        - Company info: {job_data.get('company_description', '')[:500]}
        - Transport time: {transport_time}
        - Resume highlights: {self.resume_data['competences'][:5]}
        
        Rules:
        - 3 paragraphs max
        - Use formal business French
        - Include position-specific keywords
        - Max 400 words<</SYS>>[/INST]"""

    def _build_resume_prompt(self, job_data: Dict) -> str:
        """Construct resume tailoring prompt"""
        key_skills = JobMatcher.extract_key_skills(job_data.get('description', ''))
        
        return f"""<s>[INST] <<SYS>>
        Adapt this resume for {job_data.get('title', 'position')} position:
        - Current resume: {self.resume_data['full_text'][:3000]}
        - Job requirements: {job_data.get('description', '')[:2000]}
        - Key skills to emphasize: {key_skills}
        - Company priorities: {job_data.get('company_description', '')[:500]}
        
        Rules:
        - Keep original format
        - Prioritize relevant experience
        - Add missing keywords naturally
        - Max 2 pages<</SYS>>[/INST]"""

# ======================
# API ENDPOINTS
# ======================
class AnalysisRequest(BaseModel):
    job_id: str
    job_data: Dict
    user_id: Optional[str] = None
    company_description: Optional[str] = None
    location: Optional[str] = None

@app.post("/analyze-job")
@limiter.limit("10/minute")
async def analyze_job(
    request: Request,
    analysis_request: AnalysisRequest,
    x_api_key: str = Header(None)
):
    """Endpoint for job analysis and document generation"""
    try:
        if x_api_key != os.getenv("API_SECRET"):
            raise HTTPException(status_code=403, detail="Invalid API key")
        
        generator = DocumentGenerator()
        job_data = analysis_request.job_data
        
        # Add additional data to job_data
        job_data.update({
            "company_description": analysis_request.company_description,
            "location": analysis_request.location
        })
        
        # Get transport analysis
        location_score = TransportAnalyzer.get_location_score(job_data.get('location', ''))
        transport_details = next((e for e in TRANSPORT_DATA 
                                if e.get('ville') == job_data.get('location')), {})
        
        return {
            "cover_letter": generator.generate_cover_letter(job_data),
            "tailored_resume": generator.tailor_resume(job_data),
            "transport_analysis": {
                "score": round(location_score, 2),
                "details": transport_details.get('temps_trajet', 'N/A'),
                "station": transport_details.get('station', 'N/A')
            },
            "status": "success",
            "job_id": analysis_request.job_id
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(500, "SWOT analysis parsing failed")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health_check():
    """Enhanced health check endpoint"""
    ollama_status = "ok" if check_ollama() else "error"
    db_status = "ok" if check_nocodb() else "error"
    
    return {
        "status": "ok",
        "services": {
            "ollama": ollama_status,
            "database": db_status
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def check_ollama() -> bool:
    try:
        return requests.get(
            f"{CONFIG['ai']['local_llm']['base_url']}/api/tags",
            timeout=5
        ).status_code == 200
    except:
        return False

def check_nocodb() -> bool:
    try:
        return requests.get(
            f"{CONFIG['nocodb']['api_url']}/api/v2/tables",
            headers={"xc-token": CONFIG["nocodb"]["api_token"]},
            timeout=5
        ).status_code == 200
    except:
        return False

# ======================
# RUNNER
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv("API_HOST", "0.0.0.0"), 
        port=int(os.getenv("API_PORT", "8000")),
        timeout_keep_alive=300
    )