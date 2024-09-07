import re
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient
from bson import ObjectId
import json
import io
from fpdf import FPDF
import os
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, modify as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configure static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Use environment variables for sensitive information
MONGODB_URI = os.getenv("MongoDBURI")

# Initialize MongoDB client
client = MongoClient(MONGODB_URI)
db = client["idea_generator"]
ideas_collection = db["ideas"]
reserved_ideas_collection = db["reserved_ideas"]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9, google_api_key=os.getenv("APIKEY"))

def generate_ideas(prompt):
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="{prompt}"
    )
    print(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(prompt)
    return response


def store_idea(idea, metadata):
    idea_doc = {
        "title": idea['title'],
        "description": idea['description'],
        "features": idea['features'],
        "impact": idea['impact'],
        "implementation_steps": idea['implementation_steps'],
        "tech_stack": idea['tech_stack'],
        "metadata": metadata
    }
    result = ideas_collection.insert_one(idea_doc)
    return str(result.inserted_id)

def reserve_idea(idea_id, user_id):
    idea = ideas_collection.find_one({"_id": ObjectId(idea_id)})
    if idea:
        reserved_idea = {
            "idea_id": idea_id,
            "user_id": user_id,
            "title": idea['title'],
            "description": idea['description'],
            "features": idea['features'],
            "impact": idea['impact'],
            "metadata": idea['metadata']
        }
        reserved_ideas_collection.insert_one(reserved_idea)
        return True
    return False

def get_reserved_ideas():
    return list(reserved_ideas_collection.find({}, {"title": 1, "description": 1}))

class ChatRequest(BaseModel):
    query: str
    idea: str
    category: str
    proficiency: str
    time_frame: str
    team_size: int
    technical_skills: List[str]
    project_goals: List[str]
    theme: Optional[str] = None


@app.post("/chat_with_idea")
def chat_with_idea(chat_request: ChatRequest):
    try:
        context = f"""
        User parameters:
        Category: {chat_request.category}
        Proficiency: {chat_request.proficiency}
        Time frame: {chat_request.time_frame}
        Team size: {chat_request.team_size}
        Technical skills: {', '.join(chat_request.technical_skills)}
        Project goals: {', '.join(chat_request.project_goals)}
        Theme: {chat_request.theme}

        Generated idea:
        {chat_request.idea}

        User query: {chat_request.query}

        Please provide a helpful response to the user's query about the generated idea, taking into account the user's parameters and the idea details. 
        Format your response as plain text without any special formatting or markdown. 
        Avoid using asterisks or other symbols for emphasis. 
        Keep your response concise and to the point.
        """

        response = generate_ideas(context)
        processed_response = process_response(response)

        return {"response": processed_response}
    except Exception as e:
        print(f"Error in chat_with_idea: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

def process_response(response):
    # Remove any asterisks
    response = response.replace('*', '')
    # Remove any markdown-style headers
    response = re.sub(r'#{1,6}\s', '', response)
    # Remove any other markdown formatting you might encounter
    # For example, removing bold and italic formatting:
    response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
    response = re.sub(r'_(.*?)_', r'\1', response)
    # Add more substitutions as needed
    return response.strip()

def create_pdf(idea):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=idea['title'], ln=1, align='C')
    pdf.multi_cell(0, 10, txt=f"Description: {idea['description']}")
    pdf.cell(200, 10, txt="Key Features:", ln=1)
    for feature in idea['features']:
        pdf.cell(200, 10, txt=f"- {feature}", ln=1)
    pdf.multi_cell(0, 10, txt=f"Potential Impact: {idea['impact']}")
    
    return pdf.output(dest='S').encode('latin-1')

class IdeaRequest(BaseModel):
    category: str
    proficiency: str
    time_frame: str
    team_size: int
    technical_skills: List[str]
    project_goals: List[str]
    theme: Optional[str] = None

class ReserveIdeaRequest(BaseModel):
    idea_id: str

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "name": "World"})

@app.get("/ideas")
def ideas(request: Request):
    return templates.TemplateResponse("ideas.html", {"request": request})

@app.get("/timeline")
def timeline(request: Request):
    return templates.TemplateResponse("timeline.html", {"request": request})

@app.get("/projects")
def projects(request: Request):
    return templates.TemplateResponse("projects.html", {"request": request})

@app.get("/resume")
def resume():
    return FileResponse("static/assets/Resume.pdf")

@app.get("/invent")
def invent(request: Request):
    return templates.TemplateResponse("invent.html", {"request": request})

@app.post("/generate_ideas")
def generate_ideas_route(idea_request: IdeaRequest):
    try:
        reserved_ideas = get_reserved_ideas()
        reserved_ideas_prompt = "\n".join([f"- {idea['title']}: {idea['description']}" for idea in reserved_ideas])

        exclude_ideas = idea_request.exclude_ideas if hasattr(idea_request, 'exclude_ideas') else []
        exclude_ideas_prompt = "\n".join([f"- {title}" for title in exclude_ideas])


        prompt = f"""
            As an innovative tech project idea generator for university students, create 3 unique and novel project ideas based on the following parameters:
            Category: {idea_request.category}
            Proficiency level: {idea_request.proficiency}
            Time available: {idea_request.time_frame}
            Team size: {idea_request.team_size}
            Technical skills: {', '.join(idea_request.technical_skills)}
            Project goals: {', '.join(idea_request.project_goals)}
            Additional context: {idea_request.theme}
            Focus on creating truly innovative, cutting-edge ideas that push the boundaries of current technology. Consider emerging trends, potential breakthroughs, and interdisciplinary approaches.
            The following ideas have already been reserved and should not be suggested again:
            {reserved_ideas_prompt}
            Additionally, do not suggest any of these previously generated ideas:
            {exclude_ideas_prompt}
            For each idea, provide:
            1. Project title (creative and catchy)
            2. Brief description (2-3 sentences, highlighting its uniqueness)
            3. Key features or components (3-5 bullet points)
            4. Potential impact and benefits
            5. Steps to implement (5-7 high-level steps)
            6. Best tech stack to be used and why (3-5 technologies with brief explanations)
            Format the output as a JSON array with 3 objects, each representing an idea. Use the following structure:
            [
            {{
                "title": "Project Title",
                "description": "Brief description of the project",
                "features": ["Feature 1", "Feature 2", "Feature 3"],
                "impact": "Description of potential impact and benefits",
                "implementation_steps": ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
                "tech_stack": [
                {{"name": "Technology 1", "reason": "Reason for using this technology"}},
                {{"name": "Technology 2", "reason": "Reason for using this technology"}},
                {{"name": "Technology 3", "reason": "Reason for using this technology"}}
                ]
            }},
            ...
            ]
            Ensure that each idea is distinct, innovative, and tailored to the specified parameters.
            """

        ideas_text = generate_ideas(prompt)
        print("Raw AI response:", ideas_text)  # Debug print

        # Try to extract JSON from the response
        json_start = ideas_text.find('[')
        json_end = ideas_text.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            ideas_json = ideas_text[json_start:json_end]
        else:
            raise ValueError("No JSON found in the response")

        ideas = json.loads(ideas_json)

        for idea in ideas:
            idea_id = store_idea(idea, {
                "category": idea_request.category,
                "proficiency": idea_request.proficiency,
                "time_frame": idea_request.time_frame,
                "technical_skills": idea_request.technical_skills,
                "team_size": idea_request.team_size,
                "project_goals": idea_request.project_goals
            })
            idea['id'] = idea_id

        return JSONResponse(content=ideas)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Problematic JSON: {ideas_json}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        print(f"Error in generate_ideas_route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/reserve_idea")
def reserve_idea_route(reserve_request: ReserveIdeaRequest):
    user_id = "example_user_id"  # Replace with actual user authentication
    success = reserve_idea(reserve_request.idea_id, user_id)
    return {"success": success}

@app.get("/download_pdf/{idea_id}")
def download_pdf(idea_id: str):
    idea = ideas_collection.find_one({"_id": ObjectId(idea_id)})
    if idea:
        pdf_content = create_pdf(idea)
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type='application/pdf',
            headers={"Content-Disposition": f"attachment; filename=Idea_{idea_id}.pdf"}
        )
    raise HTTPException(status_code=404, detail="Idea not found")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
