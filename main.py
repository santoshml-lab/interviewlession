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
