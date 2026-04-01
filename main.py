from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import requests
from bs4 import BeautifulSoup
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class AnalyzeRequest(BaseModel):
    company: str
    quarter: str

def scrape_transcript(company: str) -> str:
    slug = company.strip().upper()
    url = f"https://www.screener.in/company/{slug}/consolidated/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        sections = soup.find_all(["p", "li", "td"])
        text = " ".join(s.get_text(strip=True) for s in sections[:60])
        return text[:8000] if text else ""
    except Exception:
        return ""

def analyze_with_groq(company: str, quarter: str, transcript: str) -> dict:
    if transcript:
        context = f"Based on this data for {company} ({quarter}):\n\n{transcript}\n\n"
    else:
        context = f"Based on your knowledge of {company}, a publicly listed Indian company, for the {quarter} period:\n\n"

    prompt = context + """
Analyze this and return ONLY a valid JSON object (no markdown, no extra text, no code fences) with this exact structure:
{
  "companyName": "Full official company name",
  "exchange": "NSE ticker symbol",
  "sector": "sector name",
  "metrics": [
    {"label": "Revenue", "value": "₹X,XXX Cr", "change": "+X.X% YoY", "positive": true},
    {"label": "Net Profit", "value": "₹X,XXX Cr", "change": "+X.X% YoY", "positive": true},
    {"label": "Deal Wins / Key Metric", "value": "value", "change": "context", "positive": true},
    {"label": "Attrition / Key Metric 2", "value": "value", "change": "context", "positive": true}
  ],
  "growthSignals": [
    "Signal 1 with specific numbers",
    "Signal 2 with specific numbers",
    "Signal 3 with specific numbers",
    "Signal 4 with specific numbers"
  ],
  "redFlags": [
    "Risk 1 with specific detail",
    "Risk 2 with specific detail",
    "Risk 3 with specific detail",
    "Risk 4 with specific detail"
  ],
  "tone": {
    "score": 7.5,
    "label": "Cautiously Optimistic",
    "keywords": ["word1", "word2", "word3", "word4"],
    "summary": "2-3 sentence management tone analysis"
  },
  "verdict": {
    "verdict": "Cautiously Optimistic",
    "summary": "3-4 sentence investor verdict with actionable insight"
  }
}

Use real financial data. Be specific with numbers. verdict must be one of: Bullish, Cautiously Optimistic, Bearish.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )

    text = response.choices[0].message.content.strip()
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        transcript = scrape_transcript(req.company)
        result = analyze_with_groq(req.company, req.quarter, transcript)
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid response. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
