from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="InterviewPilot AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class InterviewRequest(BaseModel):
    role: str
    company: str
    difficulty: str
    question: str = ""
    answer: str = ""
    action: str


@app.post("/interview")
async def interview(req: InterviewRequest):

    if req.action == "question":

        prompt = f"""
Generate ONE realistic {req.difficulty} interview question.

Role: {req.role}
Company: {req.company}

Return only the question.
"""

    else:

        prompt = f"""
You are an expert interviewer.

Role: {req.role}
Company: {req.company}

Question:
{req.question}

Candidate Answer:
{req.answer}

Evaluate using this format:

Score: /10

Strengths:
-

Weaknesses:
-

Improved Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    return {
        "response": response.choices[0].message.content
    }

# ==========================================
# Resume Analyzer API
# Add this to main.py
# ==========================================

from fastapi import UploadFile, File, Form
import pdfplumber

@app.post("/resume")
async def resume_analyzer(

    resume: UploadFile = File(...),
    role: str = Form(...)

):

    # Read PDF
    with pdfplumber.open(resume.file) as pdf:

        text = ""

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    prompt = f"""
You are an expert ATS Resume Reviewer.

Target Job Role:
{role}

Resume:
{text}

Analyze the resume and return Markdown format.

Include:

# ATS Score (/100)

# Resume Summary

# Strengths

# Weaknesses

# Missing Skills

# Improvement Suggestions

# Interview Questions (5)

Be professional and detailed.
"""

    completion = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    return {

        "response":
        completion.choices[0].message.content

    }

# ==========================================
# Coding Interview API
# Add to main.py
# ==========================================

from pydantic import BaseModel

class CodingRequest(BaseModel):
    action: str
    language: str
    difficulty: str
    topic: str
    question: str = ""
    code: str = ""


@app.post("/coding")
async def coding_interview(req: CodingRequest):

    if req.action == "question":

        prompt = f"""
You are an expert coding interviewer.

Generate ONE coding interview question.

Language: {req.language}
Difficulty: {req.difficulty}
Topic: {req.topic}

Return in Markdown.

Include:

# Problem

# Input

# Output

# Constraints

# Example
"""

    else:

        prompt = f"""
You are an expert coding interviewer.

Evaluate this solution.

Language:
{req.language}

Difficulty:
{req.difficulty}

Topic:
{req.topic}

Question:
{req.question}

Candidate Code:
{req.code}

Return Markdown.

Include:

# Score (/10)

# Correctness

# Code Quality

# Time Complexity

# Space Complexity

# Bugs (if any)

# Optimized Solution

# Final Feedback
"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "response": completion.choices[0].message.content
    }

# ==========================================
# Performance Analyzer API
# Add to main.py
# ==========================================

@app.get("/performance")
async def performance():

    prompt = """
You are an AI Career Coach.

Generate a realistic interview performance report.

Return ONLY valid JSON.

{
  "overall_score":"8.8/10",
  "mock_interviews":"12",
  "coding_interviews":"18",
  "resume_analysis":"6",
  "interview_score":"88%",
  "coding_score":"91%",
  "ats_score":"85%",
  "strengths":"## Strengths\n- Excellent problem solving\n- Strong Python knowledge\n- Good communication skills",
  "weak_areas":"## Weak Areas\n- Dynamic Programming\n- System Design\n- Behavioral storytelling",
  "recommendation":"## AI Recommendation\nPractice medium-level DSA questions, improve system design fundamentals, and attend one mock HR interview every week."
}
"""

    completion = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={"type": "json_object"}

    )

    import json

    return json.loads(
        completion.choices[0].message.content
    )

from supabase import create_client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY
)

from pydantic import BaseModel

class SaveHistoryRequest(BaseModel):
    user_id: str
    company: str
    role: str
    score: str
    feedback: str
    interview_type: str


@app.post("/history")
async def save_history(req: SaveHistoryRequest):

    data = supabase.table("interview_history").insert({

        "user_id": req.user_id,
        "company": req.company,
        "role": req.role,
        "score": req.score,
        "feedback": req.feedback,
        "interview_type": req.interview_type

    }).execute()

    return {
        "success": True,
        "data": data.data
    }

@app.get("/history/{user_id}")
async def get_history(user_id: str):

    data = supabase.table("interview_history")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()

    return data.data

@app.delete("/history/{history_id}")
async def delete_history(history_id: int):

    supabase.table("interview_history")\
        .delete()\
        .eq("id", history_id)\
        .execute()

    return {
        "success": True
    }








